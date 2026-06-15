"""Mlflow-side proof helpers wrapping the ar-io-proof verification kernel.

Every byte-level primitive — JCS canonicalization, SHA-256, Ed25519 sign/verify,
the profile-conditional ``_*`` strip — lives in the shared ``ario_proof``
kernel (PyPI ``ar-io-proof``, conformance-gated against ``test-vectors-v1.0``).
This module is mlflow's adapter: it owns key persistence (file format, env-var
loading, auto-generate) and preserves the ``ProofEngine.create_commitment`` /
``ProofEngine.verify_commitment`` dict shape that the rest of the plugin
(``client.py``, ``model.py``, ``anchoring.py``, ``verify.py``) and its test
surface depend on.

Lenience that does NOT survive the migration (all real bugs the kernel
catches and mlflow used to mask):

- The legacy in-tree code stripped underscore-prefixed (``_*``) annotation
  keys from the signed scope *regardless of profile*. The kernel correctly
  strips them only for the mlflow profile and pre-``spec_version`` legacy
  envelopes — an ``ario.agent/v1`` envelope with an injected ``_*`` key now
  fails signature verification, the same way any other unsigned-field
  injection does. No production caller depended on the old lenience for
  agent envelopes (the cross-product test mints clean envelopes).
- The legacy code did not re-check inline ``payload`` against ``payload_hash``.
  The kernel does (for envelopes that carry inline ``payload``), which is a
  pure tightening — mlflow profiles bind externally and never carry inline
  payload, so this affects only ingested envelopes from other profiles.

Lenience that IS preserved (legitimate backward-compatibility):

- ``verify_commitment`` passes ``allow_legacy=True`` so envelopes anchored
  before the ``spec_version`` field existed still verify and surface as
  ``spec_version_status="legacy"``. The historical surface stays intact.
"""

import base64
import json
import os
import uuid
from datetime import datetime, timezone

from nacl.signing import SigningKey, VerifyKey

from ario_proof import (
    ACCEPTED_SPEC_VERSIONS as _KERNEL_ACCEPTED_SPEC_VERSIONS,
)
from ario_proof import (
    canonical_json,
    normalize_floats,
    sha256_hex,
)
from ario_proof import sign_envelope as _kernel_sign_envelope
from ario_proof import verify_envelope as _kernel_verify_envelope

__all__ = [
    "SPEC_VERSION",
    "ACCEPTED_SPEC_VERSIONS",
    "ProofEngine",
    "canonical_json",
    "normalize_floats",
    "hash_data",
    "generate_keypair",
    "load_signing_key",
    "load_signing_key_from_env",
    "load_verify_key",
]


# Envelope spec version this build emits.
SPEC_VERSION = "ario.mlflow/v1"

# Spec versions this build accepts during verification. Re-exported from the
# kernel so the mlflow-side ``ACCEPTED_SPEC_VERSIONS`` import that callers
# rely on stays valid; the kernel is the single source of truth for the
# accepted-major registry.
ACCEPTED_SPEC_VERSIONS = _KERNEL_ACCEPTED_SPEC_VERSIONS


def hash_data(data: bytes) -> str:
    """SHA-256 hex digest. Delegates to the kernel's ``sha256_hex``."""
    return sha256_hex(data)


# ---------------------------------------------------------------------------
# Key lifecycle — file format, env loading, auto-generate
#
# The kernel deliberately stays out of key lifecycle (producers own key
# storage and rotation; ``ario_proof.sign`` takes a ``SigningKey`` and signs
# bytes). These helpers keep mlflow's existing on-disk key format
# (``{"seed": base64}`` for private, ``{"key": base64}`` for public) and
# auto-generation behavior unchanged.
# ---------------------------------------------------------------------------


def generate_keypair(private_path: str, public_path: str) -> tuple[SigningKey, VerifyKey]:
    """Generate Ed25519 keypair and save to JSON files."""
    os.makedirs(os.path.dirname(private_path), exist_ok=True)
    sk = SigningKey.generate()
    vk = sk.verify_key
    with open(private_path, "w") as f:
        json.dump({"seed": base64.b64encode(bytes(sk)).decode()}, f)
    with open(public_path, "w") as f:
        json.dump({"key": base64.b64encode(bytes(vk)).decode()}, f)
    return sk, vk


def load_signing_key(path: str) -> SigningKey:
    with open(path, "r") as f:
        data = json.load(f)
    return SigningKey(base64.b64decode(data["seed"]))


def load_signing_key_from_env(env_var: str = "ARIO_MLFLOW_SIGNING_KEY") -> SigningKey | None:
    """Load signing key from base64-encoded environment variable."""
    val = os.environ.get(env_var)
    if val:
        return SigningKey(base64.b64decode(val))
    return None


def load_verify_key(path: str) -> VerifyKey:
    with open(path, "r") as f:
        data = json.load(f)
    return VerifyKey(base64.b64decode(data["key"]))


class ProofEngine:
    """Creates and verifies hash-chained, Ed25519-signed proof envelopes.

    Thin adapter over the kernel: ``create_commitment`` delegates to
    ``ario_proof.sign_envelope`` and ``verify_commitment`` to
    ``ario_proof.verify_envelope``. The class continues to own key persistence
    and translates the kernel's ``VerificationResult`` dataclass into the
    historical mlflow dict shape so the rest of the plugin's call sites and
    test fixtures keep working without churn.
    """

    def __init__(self, private_key_path: str | None = None, public_key_path: str | None = None):
        sk = load_signing_key_from_env()
        if sk:
            self._sk = sk
            self._vk = sk.verify_key
        elif private_key_path and os.path.exists(private_key_path):
            self._sk = load_signing_key(private_key_path)
            self._vk = load_verify_key(public_key_path)
        else:
            priv = private_key_path or os.path.expanduser("~/.ario-mlflow/keys/ed25519_private.json")
            pub = public_key_path or os.path.expanduser("~/.ario-mlflow/keys/ed25519_public.json")
            self._sk, self._vk = generate_keypair(priv, pub)

    def create_commitment(
        self,
        *,
        event_type: str,
        subject: dict,
        payload_bytes: bytes,
        previous_hash: str,
        event_id: str | None = None,
        signed_at: str | None = None,
    ) -> dict:
        """Create a pure-commitment proof envelope (~300 bytes on Arweave).

        Args:
            event_type: One of ``"training_complete"``, ``"model_registered"``,
                ``"prediction"``.
            subject: Identifies the source of the canonical bytes — e.g.
                ``{"type": "mlflow_run", "run_id": "..."}``.
            payload_bytes: The exact canonical bytes that were committed to
                (caller produces these via :func:`canonical_json`). The SHA-256
                hex digest becomes ``payload_hash`` — external-commitment
                binding (envelope-spec §3.1), so ``payload`` itself is NOT
                embedded.
            previous_hash: Hash of the predecessor in the chain, or
                ``"GENESIS"``.
            event_id: Optional caller-provided UUID; auto-generated if omitted.
            signed_at: Optional ISO8601 timestamp; current UTC if omitted.

        Returns:
            The signed envelope, ready to upload. Includes ``public_key``
            (derived from the signing key) and ``signature`` (Ed25519 over
            the JCS-canonicalized signed scope, per
            ``ario_proof.envelope_for_signature``).
        """
        envelope = {
            "event_id": event_id or str(uuid.uuid4()),
            "event_type": event_type,
            "subject": subject,
            "payload_hash": hash_data(payload_bytes),
            "previous_hash": previous_hash,
            "signed_at": signed_at or datetime.now(timezone.utc).isoformat(),
            "spec_version": SPEC_VERSION,
        }
        return _kernel_sign_envelope(envelope, self._sk)

    def verify_commitment(
        self,
        envelope: dict,
        payload_bytes: bytes | None = None,
    ) -> dict:
        """Verify a pure-commitment proof envelope.

        Delegates to ``ario_proof.verify_envelope`` and translates the
        ``VerificationResult`` dataclass into the dict shape mlflow consumers
        (``verify.verify_signature``, ``test_plugin_smoke``,
        ``test_plugin_verify``, the CLI report) expect.

        Args:
            envelope: The signed envelope.
            payload_bytes: Optional bytes to hash and compare to
                ``envelope["payload_hash"]`` — check 2 of the four-check flow.

        Returns:
            Dict with: ``signature_valid``, ``payload_hash_valid``,
            ``computed_payload_hash``, ``stored_payload_hash``,
            ``spec_version_status`` (one of ``"supported"``, ``"legacy"``,
            ``"unsupported"``), ``legacy_envelope``, ``overall``, and
            ``reason`` (only when ``spec_version_status == "unsupported"``).
        """
        # ``allow_legacy=True`` preserves the historical behavior of accepting
        # envelopes anchored before ``spec_version`` shipped (the only mlflow
        # envelopes on Arweave that lack the field). The kernel handles
        # non-dict input directly (returns a fully-failed VerificationResult
        # with ``legacy_envelope=False``), so pass it through unchanged
        # instead of coercing — coercing non-dict → ``{}`` would land in the
        # kernel's empty-dict-is-legacy branch and produce ``legacy_envelope
        # =True``, which misclassifies malformed input as a legacy envelope.
        result = _kernel_verify_envelope(
            envelope,
            payload_bytes=payload_bytes,
            allow_legacy=True,
        )

        is_dict = isinstance(envelope, dict)
        spec_version = envelope.get("spec_version") if is_dict else None

        # Synthesize the trichotomy from the kernel's binary signal + legacy
        # flag. ``spec_version_status`` is part of the mlflow result contract;
        # callers (verify_signature, tests, the CLI report) branch on it.
        # Non-dict input is malformed, not legacy: label it ``unsupported``
        # so the result reads coherently (e.g. ``spec_status=="unsupported"``
        # never appears alongside ``legacy_envelope=True``).
        if not is_dict:
            spec_status = "unsupported"
        elif spec_version is None:
            spec_status = "legacy"
        elif result.spec_version_ok:
            spec_status = "supported"
        else:
            spec_status = "unsupported"

        computed = sha256_hex(payload_bytes) if payload_bytes is not None else None

        out: dict = {
            "signature_valid": result.signature_ok,
            "payload_hash_valid": result.payload_hash_ok,
            "computed_payload_hash": computed,
            "stored_payload_hash": envelope.get("payload_hash") if is_dict else None,
            "spec_version_status": spec_status,
            "legacy_envelope": result.legacy_envelope,
            "overall": result.ok,
        }
        if spec_status == "unsupported":
            out["reason"] = "unsupported_spec_version" if is_dict else "envelope_not_a_json_object"
        return out
