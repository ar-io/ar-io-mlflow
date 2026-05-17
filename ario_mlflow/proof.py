"""Ed25519 signing, canonical JSON, and SHA-256 hashing for proof records."""

import base64
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone

import jcs
from nacl.signing import SigningKey, VerifyKey


# Envelope spec version this build emits. Bumped when the signed-body
# shape changes in a way that requires verifiers to know the new shape.
SPEC_VERSION = "ario.mlflow/v1"

# Spec versions this build *accepts* during verification. Includes the
# sister agent's major (``ario.agent/v1``) so plugin verifiers can check
# envelopes minted by ar-io-agent — the two share envelope spec + crypto.
# A future spec bump adds the new major here; old verifiers that don't
# know it return ``unsupported_spec_version`` and refuse to verify.
ACCEPTED_SPEC_VERSIONS = frozenset({
    "ario.mlflow/v1",
    "ario.agent/v1",
})


def _classify_spec_version(envelope: dict) -> tuple[str, str | None]:
    """Classify an envelope's ``spec_version`` field.

    Returns ``(status, reason)`` where ``status`` is one of:

    - ``"supported"`` — present and in :data:`ACCEPTED_SPEC_VERSIONS`.
    - ``"legacy"`` — field absent. Envelopes anchored before this build
      had no ``spec_version`` field. They MUST still verify; callers
      surface ``legacy_envelope=True`` so consumers can distinguish.
    - ``"unsupported"`` — present but not in the accepted set. The
      envelope advertises a spec this build doesn't know how to verify.

    ``reason`` is set to ``"unsupported_spec_version"`` only for the
    unsupported case; otherwise ``None``.
    """
    version = envelope.get("spec_version")
    if version is None:
        return ("legacy", None)
    if version in ACCEPTED_SPEC_VERSIONS:
        return ("supported", None)
    return ("unsupported", "unsupported_spec_version")


def normalize_floats(obj, precision=6):
    """Recursively round floats. Use BEFORE canonical_json when hashing values
    that may differ at floating-point precision across measurements (e.g.
    metrics re-derived from MLflow). Not applied automatically — strict JCS
    serializes the actual float value, so rounding is the caller's choice.
    """
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, dict):
        return {k: normalize_floats(v, precision) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize_floats(v, precision) for v in obj]
    return obj


def canonical_json(obj) -> bytes:
    """RFC-8785 (JSON Canonicalization Scheme) serialization.

    Strict JCS — produces deterministic UTF-8 bytes that any RFC-8785
    verifier in any language can reproduce without depending on this
    function. Matches AgentSystems Notary, Sigstore, and the broader
    RFC-8785 ecosystem. See https://www.rfc-editor.org/rfc/rfc8785.

    Numbers are serialized per ECMA-262 Number.prototype.toString (no
    trailing zeros, scientific notation for very large / small values).
    Floats are NOT pre-rounded — callers that need precision-controlled
    hashing should call ``normalize_floats`` first.
    """
    return jcs.canonicalize(obj)


def hash_data(data: bytes) -> str:
    """SHA-256 hex digest."""
    return hashlib.sha256(data).hexdigest()


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
    """Creates and verifies hash-chained, Ed25519-signed proof envelopes."""

    def __init__(self, private_key_path: str | None = None, public_key_path: str | None = None):
        # Try env var first, then key files, then auto-generate
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

    # ------------------------------------------------------------------
    # Pure-commitment envelope (~300 bytes on Arweave).
    #
    # Used by the plugin's headline API: ``anchor()``,
    # ``ArioMlflowClient``, ``VerifiedModel``. Phase 2.E deleted the
    # legacy ``create_proof`` / ``verify_local`` methods that produced
    # the v1 record-bearing envelope shape.
    # ------------------------------------------------------------------

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
        """Create a pure-commitment proof envelope.

        Args:
            event_type: One of ``"training_complete"``, ``"model_registered"``,
                ``"prediction"``.
            subject: Identifies the source of the canonical bytes — e.g.
                ``{"type": "mlflow_run", "run_id": "..."}`` or
                ``{"type": "mlflow_model_version", "name": "...",
                "version": "..."}``. Verifiers use this to find the source
                data when re-deriving the commitment.
            payload_bytes: The exact canonical bytes that were committed to
                (caller produces these via :func:`canonical_json` of whatever
                it wants to commit). The SHA-256 hex digest becomes
                ``payload_hash``.
            previous_hash: Hash of the predecessor in the chain, or
                ``"GENESIS"``. For training proofs, the prior training
                proof's ``payload_hash`` of the same registered model. For
                registration, the source run's ``ario.training_tx``. For
                predictions, the model version's ``ario.registration_tx``.
            event_id: Optional caller-provided UUID; auto-generated if
                omitted.
            signed_at: Optional ISO8601 timestamp; current UTC if omitted.

        Returns:
            The signed envelope: ``event_id``, ``event_type``, ``subject``,
            ``payload_hash``, ``previous_hash``, ``signed_at``,
            ``public_key``, ``signature``. ~300 bytes.

        Note:
            The signature covers the full envelope minus the signature
            field itself (including ``public_key``). A verifier with no
            external trust anchor can confirm only "the holder of the
            private key matching ``public_key`` produced this signature";
            trust in *whose* key it is must come from out of band.
        """
        envelope = {
            "event_id": event_id or str(uuid.uuid4()),
            "event_type": event_type,
            "subject": subject,
            "payload_hash": hash_data(payload_bytes),
            "previous_hash": previous_hash,
            "signed_at": signed_at or datetime.now(timezone.utc).isoformat(),
            "public_key": bytes(self._vk).hex(),
            "spec_version": SPEC_VERSION,
        }
        signed = self._sk.sign(canonical_json(envelope))
        envelope["signature"] = signed.signature.hex()
        return envelope

    def verify_commitment(
        self,
        envelope: dict,
        payload_bytes: bytes | None = None,
    ) -> dict:
        """Verify a pure-commitment proof envelope.

        Always checks the signature. If ``payload_bytes`` is provided,
        also re-hashes them and compares to ``envelope["payload_hash"]``
        — this is check 2 of the four-check verification flow. To run
        check 3 (live MLflow vs. anchored payload), the caller re-derives
        canonical bytes from current MLflow state and passes them here.

        Args:
            envelope: The signed envelope.
            payload_bytes: Optional bytes to hash and compare. If
                ``None``, only the signature is checked.

        Returns:
            ``signature_valid`` (bool); ``payload_hash_valid`` (bool or
            ``None`` when not checked); ``computed_payload_hash`` (str
            or ``None``); ``stored_payload_hash`` (str);
            ``spec_version_status`` (one of ``"supported"``,
            ``"legacy"``, ``"unsupported"``); ``legacy_envelope`` (bool
            — true when the envelope predates the ``spec_version``
            field); ``overall`` (bool — ``True`` only when signature
            valid *and* hash valid if checked *and* spec is not
            unsupported); ``reason`` (only set when spec is
            unsupported).
        """
        spec_status, spec_reason = _classify_spec_version(envelope)
        sig_valid = False
        try:
            # Reconstruct the signed body. Strip:
            # - ``signature`` itself (it was added after signing).
            # - Any caller-attached annotation keys (underscore-prefixed,
            #   e.g. ``_tx_id`` injected so verify_ario_attestation can
            #   route the call). By convention, ``_*`` keys are
            #   out-of-band routing metadata, not part of the signed
            #   protocol. Without this, full_verify would falsely fail
            #   the signature check when called with an envelope that
            #   any other check needs to annotate.
            body = {
                k: v for k, v in envelope.items()
                if k != "signature" and not k.startswith("_")
            }
            vk = VerifyKey(bytes.fromhex(envelope["public_key"]))
            vk.verify(canonical_json(body), bytes.fromhex(envelope["signature"]))
            sig_valid = True
        except Exception:  # noqa: BLE001 — any verification failure (decode, key shape, signature mismatch) keeps sig_valid=False; verifier must not crash on adversarial input
            pass

        payload_hash_valid: bool | None = None
        computed_hash: str | None = None
        if payload_bytes is not None:
            computed_hash = hash_data(payload_bytes)
            payload_hash_valid = computed_hash == envelope.get("payload_hash")

        # overall is True only if signature is valid AND (payload not
        # checked OR payload check passed) AND the spec_version is not
        # explicitly unsupported. Absent (legacy) and supported both
        # pass; only an envelope that advertises a spec major this
        # build doesn't know about is a hard fail.
        overall = (
            sig_valid
            and (payload_hash_valid is not False)
            and spec_status != "unsupported"
        )

        result = {
            "signature_valid": sig_valid,
            "payload_hash_valid": payload_hash_valid,
            "computed_payload_hash": computed_hash,
            "stored_payload_hash": envelope.get("payload_hash"),
            "spec_version_status": spec_status,
            "legacy_envelope": spec_status == "legacy",
            "overall": overall,
        }
        if spec_reason is not None:
            result["reason"] = spec_reason
        return result
