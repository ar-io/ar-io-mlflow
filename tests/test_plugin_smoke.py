"""Smoke tests for the ar-io-mlflow plugin.

Covers CodeRabbit PR #3 fixes and the S1 CLI write-back behaviours. No network
or MLflow server required.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.arweave import ArweaveAnchor
from ario_mlflow.verify import ArioVerifyClient
from ario_mlflow.report import generate_verification_html


# --- proof engine ---------------------------------------------------------


def test_canonical_json_and_hash_are_deterministic():
    a = {"b": 2, "a": 1, "c": [3, 1, 2]}
    b = {"a": 1, "c": [3, 1, 2], "b": 2}
    assert canonical_json(a) == canonical_json(b)
    assert hash_data(canonical_json(a)) == hash_data(canonical_json(b))


def test_canonical_json_emits_utf8_not_ascii_escapes():
    """RFC-8785 emits UTF-8 directly — non-ASCII characters are not \\uXXXX-escaped.

    Distinguishes strict JCS from Python's default json.dumps with
    ensure_ascii=True. An external JCS verifier in any language must produce
    the same UTF-8 bytes.
    """
    out = canonical_json({"name": "café"})
    assert out == b'{"name":"caf\xc3\xa9"}'


def test_canonical_json_sorts_by_utf16_code_units():
    """RFC-8785 sorts keys by UTF-16 code units, not Unicode code points.

    For the BMP characters here both orderings agree, but the test pins the
    contract — ö (U+00F6) must come before ü (U+00FC).
    """
    out = canonical_json({"ü": 1, "ö": 2})
    assert out == b'{"\xc3\xb6":2,"\xc3\xbc":1}'


def test_canonical_json_serializes_numbers_per_ecma_262():
    """RFC-8785 numbers follow ECMA-262 Number.prototype.toString.

    Vector adapted from RFC-8785 Appendix B: trailing zeros stripped, very
    large / very small values use scientific notation.
    """
    out = canonical_json({"numbers": [333333333.33333329, 1e30, 4.50, 2e-3]})
    assert out == b'{"numbers":[333333333.3333333,1e+30,4.5,0.002]}'


def test_verify_helpers_reexported_from_top_level():
    """Per plan Part 4 plugin change item 6: external users can import
    the verify helpers without spelunking ario_mlflow.verify."""
    import ario_mlflow

    # Each helper resolves through the top-level package
    assert callable(ario_mlflow.verify_signature)
    assert callable(ario_mlflow.verify_anchored_bytes)
    assert callable(ario_mlflow.verify_source_of_truth)
    assert callable(ario_mlflow.verify_ario_attestation)
    assert callable(ario_mlflow.full_verify)
    # ArioVerifyClient too
    assert isinstance(ario_mlflow.ArioVerifyClient, type)


def test_canonical_json_does_not_round_floats_implicitly():
    """Strict JCS serializes the exact float — callers that want rounded
    behaviour must call ``normalize_floats`` first. Pin this contract so a
    future regression that re-introduces silent rounding fails loudly.
    """
    from ario_mlflow.proof import normalize_floats

    raw = {"accuracy": 0.91234567}
    rounded = normalize_floats(raw, precision=6)
    assert canonical_json(raw) == b'{"accuracy":0.91234567}'
    assert canonical_json(rounded) == b'{"accuracy":0.912346}'


# Phase 2.E removed the legacy create_proof / verify_local round-trip
# tests. Their replacements for the new pure-commitment envelope shape
# follow below.


# --- pure-commitment envelope (new shape) ---------------------------------


def test_create_commitment_produces_minimal_envelope(tmp_path):
    """Envelope contains only commitment fields — no source data."""
    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    payload = canonical_json({"params": {"lr": 0.01}, "metrics": {"acc": 0.91}})
    env = engine.create_commitment(
        event_type="training_complete",
        subject={"type": "mlflow_run", "run_id": "abc123"},
        payload_bytes=payload,
        previous_hash="GENESIS",
    )
    expected_keys = {
        "event_id", "event_type", "subject",
        "payload_hash", "previous_hash", "signed_at",
        "public_key", "spec_version", "signature",
    }
    assert set(env.keys()) == expected_keys
    # No source data in the envelope — only the hash.
    assert "params" not in env
    assert "metrics" not in env
    assert env["payload_hash"] == hash_data(payload)
    # spec_version pins the envelope shape so verifiers can know what
    # they're looking at. See test_create_commitment_emits_spec_version.
    assert env["spec_version"] == "ario.mlflow/v1"


def test_create_commitment_envelope_is_small(tmp_path):
    """Sanity check: pure-commitment envelopes are bounded.

    Actual size is ~500–550 bytes — dominated by hex-encoded Ed25519
    public_key (64) and signature (128), plus a UUID, ISO timestamp,
    subject, and JSON syntax. Still an order of magnitude smaller than
    the legacy record-bearing envelope and well within Turbo's free tier.
    The plan estimated ~300 bytes; the real number is the test below.
    """
    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    env = engine.create_commitment(
        event_type="prediction",
        subject={"type": "mlflow_decision", "decision_id": "d-1"},
        payload_bytes=b'{"input_hash":"abc","output_hash":"def"}',
        previous_hash="GENESIS",
    )
    serialized = canonical_json(env)
    assert 400 < len(serialized) < 700, f"envelope was {len(serialized)} bytes"


def test_verify_commitment_signature_only(tmp_path):
    """Without payload_bytes, only the signature is checked."""
    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    env = engine.create_commitment(
        event_type="training_complete",
        subject={"type": "mlflow_run", "run_id": "r"},
        payload_bytes=b"hello",
        previous_hash="GENESIS",
    )
    result = engine.verify_commitment(env)
    assert result["signature_valid"] is True
    assert result["payload_hash_valid"] is None
    assert result["overall"] is True


def test_verify_commitment_with_matching_payload(tmp_path):
    """When payload_bytes match, both checks pass."""
    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    payload = b'{"k":"v"}'
    env = engine.create_commitment(
        event_type="prediction",
        subject={"type": "mlflow_decision", "decision_id": "d"},
        payload_bytes=payload,
        previous_hash="GENESIS",
    )
    result = engine.verify_commitment(env, payload_bytes=payload)
    assert result["signature_valid"] is True
    assert result["payload_hash_valid"] is True
    assert result["overall"] is True


def test_verify_commitment_detects_tampered_payload(tmp_path):
    """When payload_bytes don't match the anchor, overall fails."""
    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    env = engine.create_commitment(
        event_type="prediction",
        subject={"type": "mlflow_decision", "decision_id": "d"},
        payload_bytes=b'{"original":"payload"}',
        previous_hash="GENESIS",
    )
    result = engine.verify_commitment(env, payload_bytes=b'{"tampered":"payload"}')
    assert result["signature_valid"] is True
    assert result["payload_hash_valid"] is False
    assert result["overall"] is False


def test_verify_commitment_detects_tampered_envelope(tmp_path):
    """Mutating any signed field invalidates the signature."""
    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    env = engine.create_commitment(
        event_type="training_complete",
        subject={"type": "mlflow_run", "run_id": "r"},
        payload_bytes=b"hello",
        previous_hash="GENESIS",
    )
    # Tamper with the previous_hash — should break signature.
    env["previous_hash"] = "FAKE-PREDECESSOR"
    result = engine.verify_commitment(env)
    assert result["signature_valid"] is False
    assert result["overall"] is False


def test_verify_commitment_signature_covers_public_key(tmp_path):
    """Public key is part of the signed body — swapping it breaks the
    signature even if the new key is well-formed.

    This documents the design: a verifier confirms 'the holder of this
    public_key signed this'. Identity binding (whose key it is) must
    come from out of band.
    """
    from nacl.signing import SigningKey as SK

    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    env = engine.create_commitment(
        event_type="training_complete",
        subject={"type": "mlflow_run", "run_id": "r"},
        payload_bytes=b"hello",
        previous_hash="GENESIS",
    )
    # Swap to a different valid public key — signature must fail.
    other_vk = SK.generate().verify_key
    env["public_key"] = bytes(other_vk).hex()
    result = engine.verify_commitment(env)
    assert result["signature_valid"] is False


def test_verify_commitment_ignores_underscore_prefixed_caller_annotations(tmp_path):
    """Caller-attached metadata keys (underscore-prefixed by convention)
    must not invalidate signature verification.

    Concrete case: ``verify_ario_attestation`` reads ``envelope["_tx_id"]``
    when set, so the four-check ``full_verify`` flow injects ``_tx_id``
    onto the envelope before running. Without this exclusion,
    ``verify_signature`` would canonicalize the *modified* envelope
    and fail the signature check, breaking ``full_verify`` for any
    caller passing an envelope through both checks.

    Convention: ``_``-prefixed keys are out-of-band routing metadata,
    not part of the signed protocol.
    """
    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    env = engine.create_commitment(
        event_type="prediction",
        subject={"type": "mlflow_decision", "decision_id": "d-1"},
        payload_bytes=b'{"input_hash":"abc","output_hash":"def"}',
        previous_hash="GENESIS",
    )

    # Without any annotation: signature verifies as expected.
    assert engine.verify_commitment(env)["signature_valid"] is True

    # Caller attaches routing metadata (e.g. for verify_ario_attestation).
    env_annotated = dict(env)
    env_annotated["_tx_id"] = "ar-tx-some-id"
    env_annotated["_other_internal_field"] = {"foo": "bar"}

    # Signature must still verify — _tx_id and _other_internal_field
    # are stripped before reconstructing the signed body.
    result = engine.verify_commitment(env_annotated)
    assert result["signature_valid"] is True, result

    # Sanity: a NON-underscore-prefixed mutation must still fail
    # verification (so we know the strip is conservative).
    env_tampered = dict(env)
    env_tampered["event_type"] = "ATTACKER"
    assert engine.verify_commitment(env_tampered)["signature_valid"] is False


@pytest.mark.parametrize("malformed", [None, [], "hello", 42, True])
def test_verify_commitment_classifies_non_dict_as_unsupported(tmp_path, malformed):
    """Adversarial / malformed input (non-dict) must classify cleanly.

    ``overall=False`` is the safety-relevant guarantee, but the label
    fields should also read coherently: a non-dict input is malformed,
    not legacy. ``spec_version_status="unsupported"`` and
    ``legacy_envelope=False`` together describe "this isn't a valid
    envelope shape," with ``reason="envelope_not_a_json_object"``
    surfaced for diagnostics. The pre-kernel in-tree code crashed with
    ``AttributeError`` on non-dict input; the kernel-backed adapter
    soft-fails uniformly.
    """
    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    result = engine.verify_commitment(malformed)

    assert result["overall"] is False
    assert result["signature_valid"] is False
    assert result["spec_version_status"] == "unsupported"
    assert result["legacy_envelope"] is False
    assert result.get("reason") == "envelope_not_a_json_object"


def test_create_commitment_event_id_and_signed_at_overrides(tmp_path):
    """Caller may provide event_id / signed_at for deterministic tests."""
    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    env = engine.create_commitment(
        event_type="training_complete",
        subject={"type": "mlflow_run", "run_id": "r"},
        payload_bytes=b"x",
        previous_hash="GENESIS",
        event_id="11111111-1111-1111-1111-111111111111",
        signed_at="2026-04-28T00:00:00+00:00",
    )
    assert env["event_id"] == "11111111-1111-1111-1111-111111111111"
    assert env["signed_at"] == "2026-04-28T00:00:00+00:00"


# --- spec_version: emit, legacy, unsupported, cross-product ----------------


def test_create_commitment_emits_spec_version(tmp_path):
    """Every plugin-minted envelope advertises ``ario.mlflow/v1``.

    Pins the constant so a future spec bump is a single, intentional
    change rather than silent drift between plugin and sister agent.
    """
    from ario_mlflow.proof import SPEC_VERSION

    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    env = engine.create_commitment(
        event_type="training_complete",
        subject={"type": "mlflow_run", "run_id": "r"},
        payload_bytes=b"x",
        previous_hash="GENESIS",
    )
    assert env["spec_version"] == "ario.mlflow/v1"
    assert SPEC_VERSION == "ario.mlflow/v1"

    # Re-verifying our own envelope must classify it as "supported"
    # (not legacy, not unsupported) and pass overall.
    result = engine.verify_commitment(env)
    assert result["spec_version_status"] == "supported"
    assert result["legacy_envelope"] is False
    assert result["overall"] is True


def test_verify_commitment_accepts_legacy_envelope_without_spec_version(tmp_path):
    """Envelopes anchored before ``spec_version`` was emitted must still verify.

    Anything already on Arweave from earlier builds has no
    ``spec_version`` field. The verifier MUST treat absence as legacy
    (continue, but flag), not as unsupported.
    """
    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    payload = b"legacy-payload"

    # Hand-craft a legacy-shape envelope (no spec_version field) and
    # sign it with the engine's key. Mirrors what would have been
    # written by a pre-spec_version build.
    body = {
        "event_id": "legacy-1",
        "event_type": "training_complete",
        "subject": {"type": "mlflow_run", "run_id": "legacy"},
        "payload_hash": hash_data(payload),
        "previous_hash": "GENESIS",
        "signed_at": "2026-01-01T00:00:00+00:00",
        "public_key": bytes(engine._vk).hex(),
    }
    signed = engine._sk.sign(canonical_json(body))
    body["signature"] = signed.signature.hex()

    result = engine.verify_commitment(body, payload_bytes=payload)
    assert result["signature_valid"] is True
    assert result["payload_hash_valid"] is True
    assert result["spec_version_status"] == "legacy"
    assert result["legacy_envelope"] is True
    assert result["overall"] is True
    assert "reason" not in result


def test_verify_commitment_rejects_unsupported_spec_version(tmp_path):
    """An envelope advertising an unknown spec major fails overall.

    The signature itself can still be cryptographically valid (the
    envelope is honestly signed) — but a verifier that doesn't know the
    spec shape can't meaningfully attest to it, so overall is False.
    """
    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))

    body = {
        "event_id": "future-1",
        "event_type": "training_complete",
        "subject": {"type": "mlflow_run", "run_id": "r"},
        "payload_hash": hash_data(b"x"),
        "previous_hash": "GENESIS",
        "signed_at": "2026-05-15T00:00:00+00:00",
        "public_key": bytes(engine._vk).hex(),
        "spec_version": "ario.mlflow/v99",
    }
    signed = engine._sk.sign(canonical_json(body))
    body["signature"] = signed.signature.hex()

    result = engine.verify_commitment(body)
    assert result["signature_valid"] is True
    assert result["spec_version_status"] == "unsupported"
    assert result["legacy_envelope"] is False
    assert result["overall"] is False
    assert result["reason"] == "unsupported_spec_version"


def test_verify_signature_wrapper_surfaces_spec_version_state(tmp_path):
    """verify.verify_signature mirrors the proof-engine spec_version contract.

    Confirms the wrapper propagates ``spec_version_status`` and
    ``legacy_envelope``, and fails ``ok`` with the conventional reason
    for unsupported majors.
    """
    from ario_mlflow.verify import verify_signature

    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))

    # Supported (plugin-minted).
    env = engine.create_commitment(
        event_type="training_complete",
        subject={"type": "mlflow_run", "run_id": "r"},
        payload_bytes=b"x",
        previous_hash="GENESIS",
    )
    out = verify_signature(env, engine)
    assert out["ok"] is True
    assert out["spec_version_status"] == "supported"
    assert out["legacy_envelope"] is False

    # Unsupported.
    bad = {
        "event_id": "f",
        "event_type": "training_complete",
        "subject": {"type": "mlflow_run", "run_id": "r"},
        "payload_hash": hash_data(b"x"),
        "previous_hash": "GENESIS",
        "signed_at": "2026-05-15T00:00:00+00:00",
        "public_key": bytes(engine._vk).hex(),
        "spec_version": "ario.future/v1",
    }
    bad["signature"] = engine._sk.sign(canonical_json(bad)).signature.hex()
    out = verify_signature(bad, engine)
    assert out["ok"] is False
    assert out["signature_valid"] is True
    assert out["spec_version_status"] == "unsupported"
    assert out["reason"] == "unsupported_spec_version"


def test_verify_commitment_rejects_underscore_injection_on_agent_envelope(tmp_path):
    """Profile-conditional ``_*`` strip: only mlflow envelopes are lenient.

    The shared kernel strips underscore-prefixed annotation keys before
    canonicalizing the signed scope ONLY for the ``ario.mlflow/v1`` profile
    and pre-``spec_version`` legacy envelopes. ``ario.agent/v1`` envelopes
    are signed strictly — an injected ``_*`` key must invalidate the
    signature, the same as any other unsigned-field tamper. This test
    pins the asymmetry so a future kernel change can't silently re-introduce
    the historical lenience on cross-product envelopes.
    """
    from nacl.signing import VerifyKey

    vector_path = Path(__file__).parent / "fixtures" / "agent-envelope-asset-registered-01.json"
    vector = json.loads(vector_path.read_text())

    envelope = dict(vector["inputs"]["envelope_pre_signature"])
    envelope["public_key"] = vector["fixed_keypair"]["ed25519_public_hex"]
    envelope["payload_hash"] = vector["expected_outputs"]["payload_hash_hex"]
    envelope["signature"] = vector["expected_outputs"]["signature_hex"]

    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))

    # Sanity: clean agent envelope verifies (the cross-product case).
    assert engine.verify_commitment(envelope)["signature_valid"] is True

    # Inject a caller-attached ``_*`` annotation onto the agent envelope.
    # Under the kernel's strict rules this must FAIL — the strip is
    # profile-conditional, so the injection lands inside the signed scope.
    tampered = dict(envelope)
    tampered["_attacker_routing"] = "EVIL"
    result = engine.verify_commitment(tampered)
    assert result["signature_valid"] is False, (
        "_* injection on an ario.agent/v1 envelope must invalidate the "
        "signature — the kernel does not strip annotation keys for that "
        "profile"
    )


def test_verify_commitment_accepts_cross_product_agent_envelope(tmp_path):
    """Plugin verifies envelopes minted by the sister ar-io-agent (Go daemon).

    Loads a real ``ario.agent/v1`` envelope from
    ``tests/fixtures/agent-envelope-asset-registered-01.json`` (a copy
    of the agent's canonical test vector, signed with a fixed test
    seed). Confirms the plugin and agent interoperate bidirectionally
    — symmetry with the agent's own test that verifies plugin
    envelopes.
    """
    from nacl.signing import VerifyKey

    vector_path = Path(__file__).parent / "fixtures" / "agent-envelope-asset-registered-01.json"
    vector = json.loads(vector_path.read_text())

    # Reconstruct the full signed envelope from the vector. The vector
    # stores the pre-signature body with ``public_key`` and
    # ``payload_hash`` omitted; the signer fills both in before
    # canonicalizing. The signed-bytes hex in expected_outputs confirms
    # both fields are present in the actual signed body.
    envelope = dict(vector["inputs"]["envelope_pre_signature"])
    envelope["public_key"] = vector["fixed_keypair"]["ed25519_public_hex"]
    envelope["payload_hash"] = vector["expected_outputs"]["payload_hash_hex"]
    envelope["signature"] = vector["expected_outputs"]["signature_hex"]

    # Sanity-check: the embedded public key actually verifies the
    # vector's signature over the published canonical bytes. If this
    # fails, the fixture itself is bad — the spec_version check below
    # would be running against a broken envelope.
    VerifyKey(bytes.fromhex(envelope["public_key"])).verify(
        bytes.fromhex(vector["expected_outputs"]["envelope_for_sig_jcs_bytes_hex"]),
        bytes.fromhex(envelope["signature"]),
    )

    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    result = engine.verify_commitment(envelope)
    assert result["signature_valid"] is True, result
    assert result["spec_version_status"] == "supported"
    assert result["legacy_envelope"] is False
    assert result["overall"] is True


# --- ArweaveAnchor wallet fallbacks (CodeRabbit #1) -----------------------


def test_build_default_tags_for_new_envelope_is_conservative():
    """Baseline tags are derivable from the envelope, non-PII, and don't
    include experiment-name / source-name / git-commit."""
    anchor = ArweaveAnchor.__new__(ArweaveAnchor)
    envelope = {
        "event_id": "ev-1",
        "event_type": "training_complete",
        "subject": {"type": "mlflow_run", "run_id": "r-1"},
        "payload_hash": "0xabc",
        "previous_hash": "GENESIS",
        "signed_at": "2026-04-28T00:00:00+00:00",
        "public_key": "PK",
        "signature": "S",
    }
    tags = anchor._build_default_tags(envelope)

    keys = {t["name"] for t in tags}
    # Required baseline
    assert {"Content-Type", "App-Name", "App-Version", "Event-Type",
            "Event-Id", "Chain-Prev"}.issubset(keys)
    # Forbidden auto-write keys (privacy / business-context leakage)
    assert "Experiment-Name" not in keys
    assert "Source-Name" not in keys
    assert "Git-Commit" not in keys
    assert "Tracking-URI" not in keys

    by_name = {t["name"]: t["value"] for t in tags}
    assert by_name["Event-Type"] == "training_complete"
    assert by_name["Event-Id"] == "ev-1"
    assert by_name["Chain-Prev"] == "GENESIS"
    assert by_name["App-Name"] == "ar-io-mlflow"


def test_build_default_tags_supports_legacy_envelope_shape():
    """Legacy record-bearing envelopes still derive tags correctly during
    Phase 1 (before the demo refactor lands)."""
    anchor = ArweaveAnchor.__new__(ArweaveAnchor)
    legacy = {
        "record": {"event_id": "ev-2", "event_type": "training_complete"},
        "record_hash": "0xrecord",
        "previous_hash": "0xprev",
    }
    tags = anchor._build_default_tags(legacy)
    by_name = {t["name"]: t["value"] for t in tags}
    assert by_name["Event-Type"] == "training_complete"
    assert by_name["Event-Id"] == "ev-2"
    assert by_name["Chain-Prev"] == "0xprev"


def test_build_default_tags_merges_extra_tags():
    """Caller-opt-in tags get merged with baseline."""
    anchor = ArweaveAnchor.__new__(ArweaveAnchor)
    envelope = {
        "event_id": "e", "event_type": "prediction",
        "payload_hash": "h", "previous_hash": "p",
    }
    tags = anchor._build_default_tags(envelope, extra_tags={
        "Model-Name": "fraud-detector",
        "Mlflow-Run-Id": "run-abc",
    })
    by_name = {t["name"]: t["value"] for t in tags}
    assert by_name["Model-Name"] == "fraud-detector"
    assert by_name["Mlflow-Run-Id"] == "run-abc"
    assert by_name["Event-Type"] == "prediction"


def test_build_default_tags_refuses_extra_tag_key_collision():
    """Caller cannot shadow baseline tag keys (would create ambiguity at
    verification time about which value is authoritative)."""
    anchor = ArweaveAnchor.__new__(ArweaveAnchor)
    envelope = {
        "event_id": "real-id", "event_type": "training_complete",
        "payload_hash": "h", "previous_hash": "p",
    }
    tags = anchor._build_default_tags(envelope, extra_tags={
        "Event-Id": "fake-id",  # attempt to shadow
        "Event-Type": "fake-type",  # attempt to shadow
        "Custom-Tag": "fine",  # not a baseline key — allowed
    })
    by_name = {t["name"]: t["value"] for t in tags}
    # Baseline values win; caller's shadow attempt is ignored.
    assert by_name["Event-Id"] == "real-id"
    assert by_name["Event-Type"] == "training_complete"
    # Non-collision extra tag is included.
    assert by_name["Custom-Tag"] == "fine"


def test_arweave_anchor_with_missing_wallet_generates_in_memory(monkeypatch):
    monkeypatch.delenv("ARIO_MLFLOW_ARWEAVE_WALLET", raising=False)
    anchor = ArweaveAnchor(wallet_path=None)
    # Either turbo_sdk is installed and we have an enabled in-memory wallet, or
    # it is absent and init silently disables. Both are valid outcomes; crucially
    # we must not crash.
    assert isinstance(anchor.enabled, bool)


def test_arweave_anchor_with_unreadable_wallet_raises(tmp_path, monkeypatch):
    """Caller-supplied wallet path that is unreadable JSON must raise
    ``WalletLoadError`` instead of silently signing with an
    auto-generated wallet under a different identity."""
    from ario_mlflow.arweave import WalletLoadError

    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    monkeypatch.delenv("ARIO_MLFLOW_ARWEAVE_WALLET", raising=False)

    with pytest.raises(WalletLoadError, match="not valid JSON"):
        ArweaveAnchor(wallet_path=str(bad))


def test_arweave_anchor_with_structurally_invalid_jwk_raises(tmp_path, monkeypatch):
    """Valid JSON but missing RSA JWK fields must raise — the operator's
    intent (use this wallet) cannot be silently overridden by an
    auto-generated substitute."""
    from ario_mlflow.arweave import WalletLoadError

    bad = tmp_path / "incomplete.json"
    bad.write_text('{"kty": "RSA"}')  # valid JSON, missing n/e/d/...
    monkeypatch.delenv("ARIO_MLFLOW_ARWEAVE_WALLET", raising=False)

    with pytest.raises(WalletLoadError, match="not a complete RSA JWK"):
        ArweaveAnchor(wallet_path=str(bad))


# --- ArioVerifyClient normalize key rename (S1 / #5) ----------------------


def test_ario_verify_normalize_returns_attestation_level():
    client = ArioVerifyClient.__new__(ArioVerifyClient)
    client.base_url = "https://example.test"
    raw = {
        "verificationId": "v-1",
        "existence": {"status": "found"},
        "level": 3,
        "links": {"dashboard": "/dash/v-1", "pdf": "https://cdn/pdf"},
        "attestation": {"gateway": "gw-1", "attestedAt": "2026-04-21T00:00:00Z"},
    }
    out = client._normalize(raw)
    assert out["attestation_level"] == 3
    assert "level" not in out
    # Relative report URL gets prefixed with base_url; already-absolute
    # PDF URL is left alone. Attested_by maps to the gateway field.
    assert out["report_url"] == "https://example.test/dash/v-1"
    assert out["pdf_url"] == "https://cdn/pdf"
    assert out["attested_by"] == "gw-1"


def test_verify_ario_attestation_below_threshold_fails():
    """Level below min_attestation_level returns ok=False with the
    actual level surfaced for the caller to display."""
    from ario_mlflow.verify import verify_ario_attestation

    # Stub a successful but low-level ar.io response.
    class _StubClient:
        enabled = True
        def submit_verification(self, tx_id):
            return {
                "attestation_level": 1,
                "attested_by": "vilenarios.com",
                "report_url": "https://example/report",
            }

    envelope = {"_tx_id": "TX-fresh"}
    out = verify_ario_attestation(envelope, _StubClient(), min_attestation_level=2)

    assert out["ok"] is False
    assert out["reason"] == "attestation_level_below_threshold"
    assert out["attestation_level"] == 1
    assert out["min_attestation_level"] == 2
    # Still surface the level + report so the user knows it's maturing.
    assert out["attested_by"] == "vilenarios.com"
    assert out["report_url"] == "https://example/report"


def test_verify_ario_attestation_at_threshold_passes():
    """Level >= threshold returns ok=True."""
    from ario_mlflow.verify import verify_ario_attestation

    class _StubClient:
        enabled = True
        def submit_verification(self, tx_id):
            return {"attestation_level": 2, "attested_by": "gw"}

    out = verify_ario_attestation(
        {"_tx_id": "TX"}, _StubClient(), min_attestation_level=2,
    )
    assert out["ok"] is True
    assert out["attestation_level"] == 2


def test_verify_ario_attestation_strict_level_3():
    """Configurable threshold lets audit users require Level 3."""
    from ario_mlflow.verify import verify_ario_attestation

    class _StubClient:
        enabled = True
        def submit_verification(self, tx_id):
            return {"attestation_level": 2, "attested_by": "gw"}

    out = verify_ario_attestation(
        {"_tx_id": "TX"}, _StubClient(), min_attestation_level=3,
    )
    assert out["ok"] is False
    assert out["attestation_level"] == 2


def test_verify_anchored_bytes_missing_required_artifact_fails():
    """For event types in _SUBJECT_TYPES_WITH_REQUIRED_ARTIFACT, missing
    payload.json must return ok=False — silent ok=None would let
    full_verify falsely pass when the witness was wiped."""
    from ario_mlflow.verify import verify_anchored_bytes

    class _StubMlflowClient:
        def get_model_version(self, name, version):
            raise Exception("not found")

    envelope = {
        "event_type": "training_complete",
        "subject": {"type": "mlflow_run", "run_id": "missing-run"},
        "payload_hash": "0xabc",
    }
    # Patch mlflow.artifacts.download_artifacts to fail
    import ario_mlflow.verify as verify_module

    out = verify_anchored_bytes(envelope, _StubMlflowClient())
    # Run-not-found path goes through download_artifacts which will
    # also fail. ok=False because mlflow_run is in the required-artifact
    # set.
    assert out["ok"] is False, out
    assert out["artifact_expected"] is True


def test_verify_anchored_bytes_legacy_subject_returns_none():
    """Legacy v1 subject types (mlflow_trace, mlflow_decision) had no
    payload artifact. Missing artifact is legitimately 'not applicable',
    not a failure — preserves graceful handling for old proofs."""
    from ario_mlflow.verify import verify_anchored_bytes

    class _StubMlflowClient:
        pass

    envelope = {
        "event_type": "prediction",
        "subject": {"type": "mlflow_trace", "trace_id": "t-old"},
        "payload_hash": "0xabc",
    }
    out = verify_anchored_bytes(envelope, _StubMlflowClient())
    assert out["ok"] is None  # not applicable, not failure
    assert out["artifact_expected"] is False


def test_compute_overall_ok_training_fails_on_none_required_check():
    """For training_complete envelopes, ok=None on signature /
    anchored_bytes / source_of_truth must fail overall — not silently
    pass on signature alone. ar.io is genuinely optional, so its None
    is fine."""
    from ario_mlflow.verify import _compute_overall_ok

    envelope = {"event_type": "training_complete"}
    sig = {"ok": True}
    bytes_check = {"ok": None}  # MLflow witness unavailable
    sot = {"ok": None}
    ario = {"ok": None}  # Optional

    overall = _compute_overall_ok(envelope, sig, bytes_check, sot, ario)
    assert overall is False, "training proof with None on required checks must fail"


def test_compute_overall_ok_training_fails_on_none_for_sot_only():
    """Even if check 2 passed, None on check 3 must still fail
    (catches the partial-refetch case)."""
    from ario_mlflow.verify import _compute_overall_ok

    envelope = {"event_type": "training_complete"}
    sig = {"ok": True}
    bytes_check = {"ok": True}
    sot = {"ok": None}  # could not fully re-derive
    ario = {"ok": None}

    overall = _compute_overall_ok(envelope, sig, bytes_check, sot, ario)
    assert overall is False


def test_compute_overall_ok_training_passes_when_all_required_green():
    """All required checks True + ar.io None (optional) = pass."""
    from ario_mlflow.verify import _compute_overall_ok

    envelope = {"event_type": "training_complete"}
    out = _compute_overall_ok(
        envelope,
        {"ok": True}, {"ok": True}, {"ok": True}, {"ok": None},
    )
    assert out is True


def test_compute_overall_ok_prediction_requires_full_verification():
    """v2 predictions require checks 1, 2, 3 to all explicitly pass —
    same strictness as training. Phase 2 enabled this by mirroring
    the canonical payload onto the trace as ``ario.payload_json`` so
    check 3 has a real second MLflow surface to compare against the
    artifact (parallel to training's params/metrics surface).

    Anything ``None`` on a required check (sig / anchored bytes /
    source of truth) is treated as a fail rather than silently green —
    a missing live MLflow surface means we can't actually confirm the
    state, full stop. ar.io stays optional.
    """
    from ario_mlflow.verify import _compute_overall_ok

    envelope = {"event_type": "prediction"}

    # All required checks pass → overall True.
    out = _compute_overall_ok(
        envelope,
        {"ok": True},   # sig
        {"ok": True},   # anchored bytes
        {"ok": True},   # source of truth (trace tag matches artifact)
        {"ok": True},   # ar.io
    )
    assert out is True

    # Source of truth None (e.g., trace pruned) → overall False.
    out = _compute_overall_ok(
        envelope,
        {"ok": True},
        {"ok": True},
        {"ok": None},   # trace gone or no payload to compare
        {"ok": True},
    )
    assert out is False, "predictions with no live surface must fail overall"

    # Anchored bytes None (legacy v1 prediction with no artifact) →
    # overall False. v1 subjects can't reach v2's bar.
    out = _compute_overall_ok(
        envelope,
        {"ok": True},
        {"ok": None},   # legacy subject, no artifact
        {"ok": None},
        {"ok": True},
    )
    assert out is False, "v1 legacy predictions cannot fully verify under v2"


def test_anchor_continues_when_upload_raises(monkeypatch, tmp_path):
    """Transient Turbo/Arweave outage must not abort the whole
    anchor() call. Tags + artifacts should still be written; the
    result simply has anchor_result=None ('signed-only')."""
    import ario_mlflow.anchoring as anchoring

    set_tags, _, _, _, _ = _make_anchor_stubs(monkeypatch)

    class _BoomAnchor:
        enabled = True
        wallet_mode = "user-configured"
        def upload_proof(self, *a, **kw):
            raise RuntimeError("Turbo outage")

    result = anchoring.anchor(
        proof_engine=anchoring.ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=_BoomAnchor(),
    )

    # anchor_result is None — degraded to signed-only.
    assert result["anchor_result"] is None
    # But the envelope was still produced + tags written.
    assert result["envelope"]["payload_hash"]
    assert "ario.payload_hash" in result["tags"]
    assert result["tags"]["ario.verify_status"] == "signed"


def test_registration_continues_when_upload_raises(monkeypatch):
    """Transient Turbo/Arweave outage during registration must degrade
    to signed-only — the registration's signed envelope, payload
    artifact, and tags must still be written. Mirrors the anchor()
    fix; CodeRabbit second-pass.
    """
    import ario_mlflow.client as client_module
    from ario_mlflow.proof import ProofEngine

    artifacts_logged: list = []
    mv_tags: dict = {}
    statuses: list = []

    class _FakeRun:
        data = type("D", (), {"tags": {
            "ario.training_tx": "TX-T",
            "ario.artifact_hash": "h",
        }})()

    class _BoomAnchor:
        enabled = True
        wallet_mode = "user-configured"
        def upload_proof(self, *a, **kw):
            raise RuntimeError("Turbo outage during registration")

    class _Client(client_module.ArioMlflowClient):
        def __init__(self):
            import tempfile as _t
            self._proof_engine = ProofEngine(
                str(_t.mkdtemp() + "/priv"),
                str(_t.mkdtemp() + "/pub"),
            )
            self._anchor = _BoomAnchor()

        def get_run(self, rid): return _FakeRun()
        def set_model_version_tag(self, name, version, key, value):
            mv_tags.setdefault((name, str(version)), {})[key] = value
        def log_artifacts(self, run_id, local_dir, artifact_path):
            snapshot: dict[str, bytes] = {}
            for root, _dirs, files in os.walk(local_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    rel = os.path.relpath(fpath, local_dir)
                    with open(fpath, "rb") as f:
                        snapshot[rel] = f.read()
            artifacts_logged.append({"run_id": run_id, "files": snapshot})
        def _record_status(self, event_type, name, version, **kw):
            statuses.append({"event_type": event_type, "name": name, "version": str(version), **kw})

    monkeypatch.setattr(client_module, "artifact_checksums", lambda *a, **kw: {})

    c = _Client()
    c._anchor_registration("fraud", "1", run_id="train-xyz", source="runs:/train-xyz/model")

    # Despite upload raising, status is "signed" (not "failed") and
    # the local artifacts + tags are intact.
    assert statuses, "no status was recorded"
    assert statuses[-1]["status"] == "signed", statuses
    # registration_payload.json + registration_proof.json must have
    # been written to MLflow even though the upload raised.
    assert artifacts_logged, "log_artifacts was not called"
    files = artifacts_logged[0]["files"]
    assert "registration_payload.json" in files
    assert "registration_proof.json" in files
    # verify_status tag set to "signed", not "anchored".
    tags = mv_tags[("fraud", "1")]
    assert tags["ario.verify_status"] == "signed"
    # Payload hash tag still written so verify can locate the artifact.
    assert "ario.payload_hash" in tags


def test_anchor_registration_rejects_mismatched_run_id():
    """When run_id and the parsed source URI run_id disagree, the
    registration must fail loudly rather than silently producing a
    proof with internally inconsistent provenance (chain link to one
    run, ``source`` field claiming another).

    Surface: the ValueError is caught by the daemon thread's outer
    except and recorded as ``status="failed"`` with the validation
    message. Caller sees the failure via ``anchor_status()``.
    """
    import ario_mlflow.client as client_module
    from ario_mlflow.proof import ProofEngine

    statuses: list = []

    class _Client(client_module.ArioMlflowClient):
        def __init__(self):
            import tempfile as _t
            self._proof_engine = ProofEngine(
                str(_t.mkdtemp() + "/priv"),
                str(_t.mkdtemp() + "/pub"),
            )
            self._anchor = type("A", (), {"enabled": False, "upload_proof": lambda *a, **k: None})()

        def get_run(self, rid):
            raise AssertionError(
                "get_run should not have been called — validation must fail "
                "before any provenance reads happen"
            )

        def set_model_version_tag(self, *a, **kw): pass
        def log_artifacts(self, *a, **kw): pass
        def _record_status(self, event_type, name, version, **kw):
            statuses.append({"event_type": event_type, "name": name, "version": str(version), **kw})

    c = _Client()
    # Mismatched: run_id="alice" but source points to a different run.
    c._anchor_registration("fraud", "1", run_id="alice", source="runs:/bob/model")

    assert statuses, "no status was recorded"
    last = statuses[-1]
    assert last["status"] == "failed", last
    # Error message must name both run IDs so the user can fix the input.
    assert "alice" in last["error"]
    assert "bob" in last["error"]


def test_anchor_registration_accepts_matching_run_id_and_source():
    """The validation must not reject the common case where run_id and
    source URI agree — that's the most-typical caller pattern."""
    import ario_mlflow.client as client_module
    from ario_mlflow.proof import ProofEngine

    captured: dict = {"get_run_called_with": None}
    statuses: list = []

    class _FakeRun:
        data = type("D", (), {"tags": {
            "ario.training_tx": "TX-T",
            "ario.artifact_hash": "h",
        }})()

    class _Client(client_module.ArioMlflowClient):
        def __init__(self):
            import tempfile as _t
            self._proof_engine = ProofEngine(
                str(_t.mkdtemp() + "/priv"),
                str(_t.mkdtemp() + "/pub"),
            )
            self._anchor = type("A", (), {"enabled": False, "upload_proof": lambda *a, **k: None})()

        def get_run(self, rid):
            captured["get_run_called_with"] = rid
            return _FakeRun()

        def set_model_version_tag(self, *a, **kw): pass
        def log_artifacts(self, *a, **kw): pass
        def _record_status(self, event_type, name, version, **kw):
            statuses.append({"event_type": event_type, "name": name, "version": str(version), **kw})

    import ario_mlflow.client as cm
    import pytest as _pt
    monkeypatch_attr = _pt.MonkeyPatch()
    monkeypatch_attr.setattr(cm, "artifact_checksums", lambda *a, **kw: {})

    try:
        c = _Client()
        # Matching: run_id="abc" and source points to the same run.
        c._anchor_registration("fraud", "1", run_id="abc", source="runs:/abc/model")

        # No "failed" status — the matching pair is accepted.
        assert statuses, "no status was recorded"
        last = statuses[-1]
        assert last["status"] in ("anchored", "signed"), last
        # And get_run was called with the agreed-on run_id.
        assert captured["get_run_called_with"] == "abc"
    finally:
        monkeypatch_attr.undo()


def test_promotion_continues_when_upload_raises(monkeypatch):
    """Same fix for promotion: upload exception degrades to signed-only,
    payload artifact is still written. Status correctly reports "signed"
    rather than the previous "failed"."""
    import ario_mlflow.client as client_module
    from ario_mlflow.proof import ProofEngine

    artifacts_logged: list = []
    mv_tags: dict = {}
    statuses: list = []

    class _FakeMV:
        run_id = "src-run"
        tags = {"ario.registration_tx": "TX-REG-9"}

    class _BoomAnchor:
        enabled = True
        wallet_mode = "user-configured"
        def upload_proof(self, *a, **kw):
            raise RuntimeError("Turbo outage during promotion")

    class _Client(client_module.ArioMlflowClient):
        def __init__(self):
            import tempfile as _t
            self._proof_engine = ProofEngine(
                str(_t.mkdtemp() + "/priv"),
                str(_t.mkdtemp() + "/pub"),
            )
            self._anchor = _BoomAnchor()

        def get_model_version(self, name, version): return _FakeMV()
        def set_model_version_tag(self, name, version, key, value):
            mv_tags.setdefault((name, str(version)), {})[key] = value
        def log_artifacts(self, run_id, local_dir, artifact_path):
            snapshot: dict[str, bytes] = {}
            for root, _dirs, files in os.walk(local_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    rel = os.path.relpath(fpath, local_dir)
                    with open(fpath, "rb") as f:
                        snapshot[rel] = f.read()
            artifacts_logged.append({"run_id": run_id, "files": snapshot})
        def _record_status(self, event_type, name, version, **kw):
            statuses.append({"event_type": event_type, "name": name, "version": str(version), **kw})

    c = _Client()
    c._anchor_promotion("fraud", "5", "Staging", "Production")

    assert statuses, "no status was recorded"
    assert statuses[-1]["status"] == "signed", statuses
    # Per Cluster 2 fix: promotion artifacts keyed by event_id under
    # promotions/<event_id>/.
    assert artifacts_logged, "log_artifacts was not called"
    files = artifacts_logged[0]["files"]
    assert any(f.startswith("promotions/") and f.endswith("payload.json") for f in files), list(files.keys())
    assert any(f.startswith("promotions/") and f.endswith("proof.json") for f in files)
    # Per CodeRabbit third-pass: ario.promotion_payload_hash must be
    # written even on signed-only fallback so model versions advertise
    # their promotion provenance regardless of upload outcome.
    tags = mv_tags.get(("fraud", "5"), {})
    assert "ario.promotion_payload_hash" in tags, tags
    # ario.promotion_tx is only written when upload actually succeeded.
    assert "ario.promotion_tx" not in tags, tags


def test_ario_verify_normalize_handles_null_sub_objects():
    """Regression for 2026-04-28 bug: ar.io Verify returns explicit
    ``null`` for ``attestation`` / ``links`` / ``existence`` sub-objects
    when a TX is too fresh to attest. ``dict.get(key, default)`` returns
    the actual ``None`` value when the key exists, not the default — so
    we must use ``or {}`` to collapse null and missing into ``{}`` before
    indexing into the sub-object.
    """
    client = ArioVerifyClient.__new__(ArioVerifyClient)
    client.base_url = "https://example.test"

    # Real-world response shape from the public ar.io Verify endpoint
    # for a freshly-anchored TX (not yet propagated through the network).
    raw = {
        "verificationId": "vrf_xyz",
        "txId": "TX-fresh",
        "level": 1,
        "existence": {"status": "not_found", "blockHeight": None},
        "authenticity": {"status": "unverified"},
        "owner": {"address": None, "publicKey": None},
        "metadata": {"dataSize": None, "tags": []},
        "bundle": {"isBundled": False},
        "gatewayAssessment": {"verified": None},
        "attestation": None,  # ← explicit null, not missing
        "links": {"dashboard": None, "pdf": None, "rawData": None},
    }

    # Must not raise on ``attestation`` being None.
    out = client._normalize(raw)
    assert out["verification_id"] == "vrf_xyz"
    assert out["attestation_level"] == 1
    assert out["status"] == "not_found"
    assert out["attested_by"] is None
    assert out["attested_at"] is None
    assert out["report_url"] is None
    assert out["pdf_url"] is None


# --- HTML report (CodeRabbit #5, #6) --------------------------------------


def _minimal_proof(tx_id: str = "TX123") -> dict:
    return {
        "record": {
            "event_type": "training_complete",
            "run_id": "run-abc",
            "timestamp": "2026-04-21T00:00:00Z",
        },
        "record_hash": "a" * 64,
        "previous_hash": "GENESIS",
        "signature": "b" * 128,
        "public_key": "c" * 64,
    }


def test_report_renders_without_crash():
    html = generate_verification_html(
        _minimal_proof(),
        anchor_result={"tx_id": "TX123", "url": "https://turbo-gateway.com/TX123", "receipt": None},
        artifact_hash="deadbeef",
    )
    assert "ar.io Verification Report" in html
    assert "TX123" in html


def test_report_curl_example_uses_raw_tx_path():
    html = generate_verification_html(
        _minimal_proof(),
        anchor_result={"tx_id": "TX123", "url": "https://turbo-gateway.com/TX123", "receipt": None},
        artifact_hash="deadbeef",
    )
    # CodeRabbit #6: fetch is /raw/<tx_id>, not /<tx_id>/raw
    assert "/raw/TX123" in html
    assert "TX123/raw" not in html


def test_report_verify_command_override_and_url_base():
    html = generate_verification_html(
        _minimal_proof(),
        anchor_result={"tx_id": "TX999", "url": "https://turbo-gateway.com/TX999", "receipt": None},
        artifact_hash="ab",
        cli_verify_cmd="ar-io-mlflow verify model foo/1",
        verify_base_url="https://custom.example/verify",
    )
    assert "ar-io-mlflow verify model foo/1" in html
    assert "https://custom.example/verify/TX999" in html
    # Old hardcoded hostname must not appear when overridden.
    assert "vilenarios.com" not in html


def test_report_verify_command_fallback_uses_run_id():
    html = generate_verification_html(
        _minimal_proof(),
        anchor_result={"tx_id": "TX5", "url": "https://turbo-gateway.com/TX5", "receipt": None},
        artifact_hash="ab",
    )
    assert "ar-io-mlflow verify run run-abc" in html


def test_report_shows_attestation_level_when_verified():
    html = generate_verification_html(
        _minimal_proof(),
        anchor_result={"tx_id": "TX1", "url": "https://turbo-gateway.com/TX1", "receipt": None},
        artifact_hash="deadbeef",
        verification={
            "attestation_level": 3,
            "report_url": "https://verify.example/v/1",
            "attested_by": "ar.io operator",
            "attested_at": "2026-04-21T00:00:00Z",
        },
    )
    assert "Level 3" in html
    assert "ar.io operator" in html
    # When verification is present, the "run CLI to verify" nudge block is hidden.
    assert "to verify this proof" not in html


# --- CLI wiring (S1) ------------------------------------------------------


def test_cli_verify_subparser_includes_trace():
    """Exercises the real ario_mlflow.cli.build_parser — not a handcrafted copy."""
    from ario_mlflow.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["verify", "trace", "trace-xyz"])
    assert args.command == "verify"
    assert args.verify_type == "trace"
    assert args.trace_id == "trace-xyz"


def test_cli_verify_run_and_model_and_audit_parse():
    """Real-parser smoke coverage for the other verify subcommands."""
    from ario_mlflow.cli import build_parser

    parser = build_parser()

    run_args = parser.parse_args(["verify", "run", "run-123"])
    assert run_args.command == "verify" and run_args.verify_type == "run"
    assert run_args.run_id == "run-123"

    model_args = parser.parse_args(["verify", "model", "foo/1"])
    assert model_args.command == "verify" and model_args.verify_type == "model"
    assert model_args.model == "foo/1"

    audit_args = parser.parse_args(["audit", "foo/1"])
    assert audit_args.command == "audit"
    assert audit_args.model == "foo/1"


# --- CLI verification-tag mapping (S1) ------------------------------------


def test_verification_run_tags_maps_all_fields():
    from ario_mlflow.cli import _verification_run_tags

    tags = _verification_run_tags({
        "attestation_level": 3,
        "report_url": "https://r/",
        "attested_by": "gw",
        "attested_at": "2026-04-21T00:00:00Z",
    })
    assert tags == {
        "ario.verify_status": "verified",
        "ario.attestation_level": "3",
        "ario.report_url": "https://r/",
        "ario.attested_by": "gw",
        "ario.attested_at": "2026-04-21T00:00:00Z",
    }


def test_verification_run_tags_skips_when_level_missing():
    from ario_mlflow.cli import _verification_run_tags

    # Attestation not yet granted — don't mark as verified.
    out = _verification_run_tags({"report_url": "https://r/"})
    assert "ario.verify_status" not in out
    assert "ario.attestation_level" not in out
    assert out == {"ario.report_url": "https://r/"}


# --- CLI NO_COLOR support -------------------------------------------------


def test_cli_glyph_includes_ansi_when_no_color_unset(monkeypatch):
    """Default behavior: ANSI escape codes wrap the glyph."""
    from ario_mlflow.cli import _check_glyph

    monkeypatch.delenv("NO_COLOR", raising=False)
    out = _check_glyph()
    assert "\033[" in out
    assert "✓" in out


def test_cli_glyph_strips_ansi_when_no_color_set(monkeypatch):
    """NO_COLOR=1 (or any non-empty value) strips ANSI escapes for clean
    output to logs, CI artifacts, and downstream parsers — community
    convention from no-color.org."""
    from ario_mlflow.cli import _check_glyph, _cross_glyph, _pending_glyph

    monkeypatch.setenv("NO_COLOR", "1")
    assert "\033[" not in _check_glyph()
    assert "\033[" not in _cross_glyph()
    assert "\033[" not in _pending_glyph()
    # Glyph characters themselves are preserved.
    assert _check_glyph() == "✓"
    assert _cross_glyph() == "✗"
    assert _pending_glyph() == "?"


def test_verification_run_tags_empty_for_none():
    from ario_mlflow.cli import _verification_run_tags

    assert _verification_run_tags(None) == {}


# --- VerifiedModel ordering: integrity must run before load_model (CodeRabbit r2 #4) ---


def test_verified_model_checks_integrity_before_load(monkeypatch):
    """Regression: tampered pyfunc artifacts must be rejected before load_model runs.

    We stub every MLflow surface VerifiedModel touches and record the order that
    artifact_checksums and mlflow.pyfunc.load_model are invoked. On mismatch,
    IntegrityError must fire before load_model is ever called.
    """
    import ario_mlflow.model as model_module
    from ario_mlflow.model import VerifiedModel, IntegrityError

    calls: list[str] = []

    class _FakeRun:
        data = type("D", (), {"tags": {"ario.artifact_hash": "EXPECTED"}})()

    class _FakeMV:
        name = "foo"
        version = 1
        run_id = "run-xyz"
        source = "runs:/run-xyz/model"

    class _FakeClient:
        def get_model_version(self, name, version):
            return _FakeMV()

        def get_run(self, run_id):
            return _FakeRun()

    def _fake_checksums(run_id, *a, **kw):
        calls.append("artifact_checksums")
        return {"model/foo": "deadbeef"}  # will hash != "EXPECTED"

    def _fake_load_model(uri):
        calls.append("load_model")
        return object()

    monkeypatch.setattr(model_module.mlflow.tracking, "MlflowClient", lambda: _FakeClient())
    monkeypatch.setattr(model_module, "artifact_checksums", _fake_checksums)
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model", _fake_load_model)

    with pytest.raises(IntegrityError):
        VerifiedModel("models:/foo/1")

    # The key assertion: load_model must NOT have been reached.
    assert "artifact_checksums" in calls, calls
    assert "load_model" not in calls, (
        "Tampered model code would have executed: load_model ran before IntegrityError"
    )


def test_verified_model_loads_only_after_integrity_passes(monkeypatch):
    """Complement: when hashes match, load_model still runs and _artifact_verified is True."""
    import ario_mlflow.model as model_module
    from ario_mlflow.model import VerifiedModel
    from ario_mlflow.proof import canonical_json, hash_data

    # Build matching hashes so verification succeeds.
    checksums = {"model/foo": "deadbeef"}
    expected = hash_data(canonical_json(checksums))

    class _FakeRun:
        data = type("D", (), {"tags": {"ario.artifact_hash": expected}})()

    class _FakeMV:
        name = "foo"
        version = 1
        run_id = "run-xyz"
        source = "runs:/run-xyz/model"

    class _FakeClient:
        def get_model_version(self, n, v): return _FakeMV()
        def get_run(self, rid): return _FakeRun()

    call_order: list[str] = []

    def _fake_checksums(run_id, *a, **kw):
        call_order.append("integrity")
        return checksums

    def _fake_load_model(uri):
        call_order.append("load")
        return object()

    monkeypatch.setattr(model_module.mlflow.tracking, "MlflowClient", lambda: _FakeClient())
    monkeypatch.setattr(model_module, "artifact_checksums", _fake_checksums)
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model", _fake_load_model)

    vm = VerifiedModel("models:/foo/1")
    assert vm._artifact_verified is True
    # Integrity ran first, load ran second.
    assert call_order == ["integrity", "load"], call_order


# --- parse_runs_uri helper -------------------------------------------------


def test_parse_runs_uri_extracts_run_and_artifact_path():
    from ario_mlflow.anchoring import parse_runs_uri

    assert parse_runs_uri("runs:/abc123/model") == ("abc123", "model")
    assert parse_runs_uri("runs:/abc123/sklearn-model") == ("abc123", "sklearn-model")
    assert parse_runs_uri("runs:/abc123/nested/path") == ("abc123", "nested/path")
    assert parse_runs_uri("runs:/abc123") == ("abc123", None)
    assert parse_runs_uri("s3://bucket/model") == (None, None)
    assert parse_runs_uri(None) == (None, None)
    assert parse_runs_uri("") == (None, None)


# --- CodeRabbit round 3 regressions ---------------------------------------


def test_verified_model_uses_mv_source_for_load_and_integrity(monkeypatch):
    """Regression for CodeRabbit r3 #3: non-default artifact paths must be respected.

    When a model is registered at e.g. ``sklearn-model``, load_uri must be
    ``mv.source`` (not ``runs:/<id>/model``) and the integrity check must hash
    the ``sklearn-model`` artifact path.
    """
    import ario_mlflow.model as model_module
    from ario_mlflow.model import VerifiedModel
    from ario_mlflow.proof import canonical_json, hash_data

    checksums = {"sklearn-model/data.pkl": "abc"}
    expected = hash_data(canonical_json(checksums))

    class _FakeRun:
        data = type("D", (), {"tags": {"ario.artifact_hash": expected}})()

    class _FakeMV:
        name = "foo"
        version = 1
        run_id = "run-xyz"
        source = "runs:/run-xyz/sklearn-model"

    class _FakeClient:
        def get_model_version(self, n, v): return _FakeMV()
        def get_run(self, rid): return _FakeRun()

    recorded: dict = {}

    def _fake_checksums(run_id, *, artifact_path="model", **_):
        recorded["artifact_path"] = artifact_path
        return checksums

    def _fake_load_model(uri):
        recorded["load_uri"] = uri
        return object()

    monkeypatch.setattr(model_module.mlflow.tracking, "MlflowClient", lambda: _FakeClient())
    monkeypatch.setattr(model_module, "artifact_checksums", _fake_checksums)
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model", _fake_load_model)

    vm = VerifiedModel("models:/foo/1")
    assert vm._artifact_verified is True
    assert recorded["artifact_path"] == "sklearn-model"
    assert recorded["load_uri"] == "runs:/run-xyz/sklearn-model"


# --- CodeRabbit round 4 regressions ---------------------------------------


def test_artifact_checksums_excludes_registration_metadata(monkeypatch, tmp_path):
    """MLflow's create_model_version adds a ``registered_model_meta``
    file to the artifact dir AFTER anchor() has already recorded the
    artifact_hash for the un-registered state. Without exclusion, a
    later VerifiedModel re-hash would falsely trip IntegrityError —
    not because anyone tampered, but because MLflow's own bookkeeping
    grew the file set. This test pins the exclusion contract.
    """
    import ario_mlflow.anchoring as anchoring
    from ario_mlflow.anchoring import artifact_checksums

    # Stage a fake artifact tree containing a registered_model_meta
    # file alongside real model files.
    fake_model_dir = tmp_path / "downloaded_model"
    fake_model_dir.mkdir()
    (fake_model_dir / "MLmodel").write_bytes(b"flavors:\n  sklearn: {}\n")
    (fake_model_dir / "model.pkl").write_bytes(b"\x80\x04\x95fake_pickle_bytes")
    (fake_model_dir / "registered_model_meta").write_bytes(
        b'{"model_name": "credit-scorer", "version": "1"}'
    )

    monkeypatch.setattr(
        anchoring.mlflow.artifacts, "download_artifacts",
        lambda run_id, artifact_path: str(fake_model_dir),
    )

    checksums = artifact_checksums("any-run-id", artifact_path="model")
    # Real model files hashed:
    assert "MLmodel" in checksums
    assert "model.pkl" in checksums
    # registration bookkeeping NOT hashed:
    assert "registered_model_meta" not in checksums


def test_artifact_checksums_raises_on_download_failure(monkeypatch):
    """Regression for CodeRabbit r4 #1: silent {} on failure would be anchored as a bogus hash."""
    import ario_mlflow.anchoring as anchoring

    def _boom(*a, **kw):
        raise RuntimeError("tracking store unavailable")

    monkeypatch.setattr(anchoring.mlflow.artifacts, "download_artifacts", _boom)

    with pytest.raises(anchoring.ArtifactAccessError) as excinfo:
        anchoring.artifact_checksums("run-123", artifact_path="model")
    assert "run-123" in str(excinfo.value)
    assert "tracking store unavailable" in str(excinfo.value)


def test_anchor_omits_artifact_hash_when_artifacts_unavailable(monkeypatch, tmp_path):
    """Regression: anchor() must not publish an empty-tree hash as ario.artifact_hash.

    Adapted for the pure-commitment redesign: artifact_hash semantics are
    unchanged (sha256 of artifact_checksums dict, used by VerifiedModel
    for its load-time integrity check), and the rule that anchor() must
    not publish a hash when artifacts are unavailable still holds. The
    payload that's now committed-to via payload_hash also reflects the
    empty checksums map honestly.
    """
    import ario_mlflow.anchoring as anchoring
    from ario_mlflow.anchoring import anchor, ArtifactAccessError

    def _boom(run_id, artifact_path="model"):
        raise ArtifactAccessError("simulated failure")

    monkeypatch.setattr(anchoring, "artifact_checksums", _boom)

    class _RunData:
        params: dict = {}
        metrics: dict = {}
        tags: dict = {}

    class _RunInfo:
        run_id = "run-xyz"

    class _ActiveRun:
        info = _RunInfo()

    # Default fake dataset input so anchor()'s fail-closed input-side
    # check passes — this test is about artifact-availability, not
    # input-side anchoring.
    class _DefaultDataset:
        name = "ds"; source = "s.csv"; source_type = "local"
        digest = "d"; schema = '{"mlflow_colspec":[]}'
    class _DefaultInputTag:
        key = "mlflow.data.context"; value = "training"
    class _DefaultDatasetInput:
        dataset = _DefaultDataset(); tags = [_DefaultInputTag()]
    class _DefaultRunInputs:
        dataset_inputs = [_DefaultDatasetInput()]

    class _FakeRun:
        data = _RunData()
        inputs = _DefaultRunInputs()

    class _FakeMlflowClient:
        def get_run(self, run_id): return _FakeRun()
        def set_tag(self, run_id, key, value):
            set_tags.setdefault(run_id, {})[key] = value
        def search_model_versions(self, query): return []
        def get_registered_model(self, name): return None
        def set_registered_model_tag(self, *a, **kw): pass

    set_tags: dict = {}

    class _FakeAnchor:
        enabled = False
        def upload_proof(self, *a, **kw): return None

    monkeypatch.setattr(anchoring.mlflow, "active_run", lambda: _ActiveRun())
    monkeypatch.setattr(anchoring.mlflow.tracking, "MlflowClient", lambda: _FakeMlflowClient())
    monkeypatch.setattr(anchoring.mlflow, "log_artifacts", lambda *a, **kw: None)
    monkeypatch.setattr(anchoring.mlflow, "get_active_trace_id", lambda: None, raising=False)
    monkeypatch.setattr(anchoring.mlflow, "get_tracking_uri", lambda: "file:./mlruns")

    result = anchor(
        proof_engine=anchoring.ProofEngine(
            str(tmp_path / "priv"), str(tmp_path / "pub")
        ),
        arweave=_FakeAnchor(),
    )

    # The fatal assertion: no ario.artifact_hash tag was written when
    # artifacts could not be hashed.
    assert "ario.artifact_hash" not in set_tags.get("run-xyz", {}), set_tags
    assert "ario.artifact_hash" not in result["tags"]
    # Payload reflects empty checksums honestly — no fabricated hash.
    assert result["payload"]["artifact_checksums"] == {}
    assert result["artifact_status"] == "hash_failed"


def test_anchor_accepts_custom_artifact_path(monkeypatch, tmp_path):
    """Regression for CodeRabbit r4 #2: anchor() hashes the logged path, not hardcoded 'model'."""
    import ario_mlflow.anchoring as anchoring
    from ario_mlflow.anchoring import anchor

    recorded: dict = {}

    def _capture_path(run_id, artifact_path="model"):
        recorded["artifact_path"] = artifact_path
        return {f"{artifact_path}/data.pkl": "abc123"}

    monkeypatch.setattr(anchoring, "artifact_checksums", _capture_path)

    class _RunData:
        params: dict = {}
        metrics: dict = {}
        tags: dict = {}

    class _RunInfo:
        run_id = "run-abc"

    class _ActiveRun:
        info = _RunInfo()

    # Default fake dataset input for anchor()'s fail-closed input check
    # (this test is about artifact path resolution, not input-side anchoring).
    class _DefaultDataset:
        name = "ds"; source = "s.csv"; source_type = "local"
        digest = "d"; schema = '{"mlflow_colspec":[]}'
    class _DefaultInputTag:
        key = "mlflow.data.context"; value = "training"
    class _DefaultDatasetInput:
        dataset = _DefaultDataset(); tags = [_DefaultInputTag()]
    class _DefaultRunInputs:
        dataset_inputs = [_DefaultDatasetInput()]

    class _FakeRun:
        data = _RunData()
        inputs = _DefaultRunInputs()

    class _FakeMlflowClient:
        def get_run(self, run_id): return _FakeRun()
        def set_tag(self, *a, **kw): pass
        def search_model_versions(self, query): return []
        def get_registered_model(self, name): return None
        def set_registered_model_tag(self, *a, **kw): pass

    class _FakeAnchor:
        enabled = False
        def upload_proof(self, *a, **kw): return None

    monkeypatch.setattr(anchoring.mlflow, "active_run", lambda: _ActiveRun())
    monkeypatch.setattr(anchoring.mlflow.tracking, "MlflowClient", lambda: _FakeMlflowClient())
    monkeypatch.setattr(anchoring.mlflow, "log_artifacts", lambda *a, **kw: None)
    monkeypatch.setattr(anchoring.mlflow, "get_active_trace_id", lambda: None, raising=False)
    monkeypatch.setattr(anchoring.mlflow, "get_tracking_uri", lambda: "file:./mlruns")

    anchor(
        proof_engine=anchoring.ProofEngine(
            str(tmp_path / "priv"), str(tmp_path / "pub")
        ),
        arweave=_FakeAnchor(),
        artifact_path="sklearn-model",
    )

    assert recorded["artifact_path"] == "sklearn-model"


# --- OTel auto-capture (Phase 1.16) ---------------------------------------
#
# These tests monkeypatch capture_otel_context directly to avoid touching
# OTel's global TracerProvider state (which doesn't reset cleanly between
# tests and corrupts the test suite). The function-level test below
# exercises the real helper without setting up OTel — it should return
# {} when no span is active, which is the default in a fresh test
# process.


def test_capture_otel_context_returns_empty_when_no_active_span():
    """No active OTel span (the default in a fresh test process) → {}.

    Exercises the real helper end-to-end. We avoid setting up a real
    OTel TracerProvider because OTel's global state doesn't reset
    cleanly and corrupts neighbouring tests.
    """
    from ario_mlflow.anchoring import capture_otel_context

    result = capture_otel_context()
    assert result == {}


def test_capture_otel_context_respects_env_var_optout(monkeypatch):
    """ARIO_MLFLOW_CAPTURE_OTEL=false suppresses capture even if a span
    might be active.

    Same env-var-only test — short-circuits before any OTel import.
    """
    from ario_mlflow.anchoring import capture_otel_context

    monkeypatch.setenv("ARIO_MLFLOW_CAPTURE_OTEL", "false")
    assert capture_otel_context() == {}

    # Other "off" spellings.
    for value in ("0", "no", "off", "FALSE", "False"):
        monkeypatch.setenv("ARIO_MLFLOW_CAPTURE_OTEL", value)
        assert capture_otel_context() == {}, f"value={value!r} should opt out"


def test_anchor_auto_captures_otel_when_helper_returns_ids(monkeypatch, tmp_path):
    """anchor() merges capture_otel_context()'s return value into the
    canonical payload by default. Verified by stubbing the helper to
    return a known dict — avoids real OTel setup."""
    import ario_mlflow.anchoring as anchoring

    *_, fake_anchor = _make_anchor_stubs(monkeypatch)
    monkeypatch.setattr(anchoring, "capture_otel_context", lambda: {
        "otel_trace_id": "aa" * 16,
        "otel_span_id": "bb" * 8,
    })

    result = anchoring.anchor(
        proof_engine=anchoring.ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=fake_anchor,
    )

    assert result["payload"]["otel_trace_id"] == "aa" * 16
    assert result["payload"]["otel_span_id"] == "bb" * 8


def test_anchor_caller_otel_metadata_wins_over_auto_capture(monkeypatch, tmp_path):
    """When both auto-capture AND caller metadata supply otel_trace_id,
    the caller's value wins. Caller is the source of truth for what
    they want signed."""
    import ario_mlflow.anchoring as anchoring

    *_, fake_anchor = _make_anchor_stubs(monkeypatch)
    monkeypatch.setattr(anchoring, "capture_otel_context", lambda: {
        "otel_trace_id": "aa" * 16,
    })

    explicit = "dd" * 16
    result = anchoring.anchor(
        proof_engine=anchoring.ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=fake_anchor,
        metadata={"otel_trace_id": explicit},
    )

    assert result["payload"]["otel_trace_id"] == explicit


def test_anchor_capture_otel_false_skips_helper_call(monkeypatch, tmp_path):
    """Per-call capture_otel=False bypasses the helper entirely — even
    if it would have returned IDs, none flow into the payload."""
    import ario_mlflow.anchoring as anchoring

    *_, fake_anchor = _make_anchor_stubs(monkeypatch)
    helper_calls: list = []

    def _helper_should_not_run():
        helper_calls.append("called")
        return {"otel_trace_id": "aa" * 16}

    monkeypatch.setattr(anchoring, "capture_otel_context", _helper_should_not_run)

    result = anchoring.anchor(
        proof_engine=anchoring.ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=fake_anchor,
        capture_otel=False,
    )

    assert helper_calls == [], "capture_otel_context should not be called"
    assert "otel_trace_id" not in result["payload"]


# --- anchor() pure-commitment behaviour (Phase 1.3) -----------------------


def _make_anchor_stubs(monkeypatch, *, run_id="run-test", run_tags=None,
                      registered_models_for_run=None, rm_tags=None,
                      anchor_enabled=False, upload_returns=None):
    """Helper to set up the MLflow stubs anchor() needs.

    Returns (set_tags_dict, rm_set_tags_dict, log_artifacts_calls,
    upload_calls) so tests can assert what anchor() did.
    """
    import ario_mlflow.anchoring as anchoring

    set_tags: dict = {}
    rm_set_tags: dict = {}
    log_artifacts_calls: list = []
    upload_calls: list = []

    class _RunData:
        params: dict = {}
        metrics: dict = {}
        tags: dict = run_tags or {}

    class _RunInfo:
        pass
    _RunInfo.run_id = run_id

    class _ActiveRun:
        info = _RunInfo()

    # Default fake dataset input so anchor()'s fail-closed check passes
    # in tests that aren't specifically about input-side anchoring.
    # Tests for input-side anchoring live in tests/test_input_anchoring.py
    # and use their own stubs.
    class _DefaultDataset:
        name = "smoke_test_dataset"
        source = "smoke.csv"
        source_type = "local"
        digest = "smoke-digest"
        schema = '{"mlflow_colspec":[]}'

    class _DefaultInputTag:
        key = "mlflow.data.context"
        value = "training"

    class _DefaultDatasetInput:
        dataset = _DefaultDataset()
        tags = [_DefaultInputTag()]

    class _DefaultRunInputs:
        dataset_inputs = [_DefaultDatasetInput()]

    class _FakeRun:
        data = _RunData()
        inputs = _DefaultRunInputs()

    class _FakeRegisteredModel:
        def __init__(self, name):
            self.name = name
            self.tags = rm_tags or {}

    class _FakeModelVersion:
        def __init__(self, name):
            self.name = name

    class _FakeMlflowClient:
        def get_run(self, rid): return _FakeRun()
        def set_tag(self, rid, key, value):
            set_tags.setdefault(rid, {})[key] = value
        def search_model_versions(self, query):
            return [_FakeModelVersion(n) for n in (registered_models_for_run or [])]
        def get_registered_model(self, name):
            return _FakeRegisteredModel(name)
        def set_registered_model_tag(self, name, key, value):
            rm_set_tags.setdefault(name, {})[key] = value

    class _FakeAnchor:
        enabled = anchor_enabled
        wallet_mode = "user-configured"
        def upload_proof(self, env, *a, **kw):
            upload_calls.append(env)
            return upload_returns

    def _capture_artifacts(local_dir, ap):
        # Snapshot file contents BEFORE the TemporaryDirectory context
        # manager cleans up. Storing paths alone is useless because
        # anchor() uses tempfile.TemporaryDirectory which is gone by the
        # time the test inspects.
        snapshot: dict[str, bytes] = {}
        for root, _dirs, files in os.walk(local_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, local_dir)
                with open(fpath, "rb") as f:
                    snapshot[rel] = f.read()
        log_artifacts_calls.append({"artifact_path": ap, "files": snapshot})

    monkeypatch.setattr(anchoring.mlflow, "active_run", lambda: _ActiveRun())
    monkeypatch.setattr(anchoring.mlflow.tracking, "MlflowClient", lambda: _FakeMlflowClient())
    monkeypatch.setattr(anchoring.mlflow, "log_artifacts", _capture_artifacts)
    monkeypatch.setattr(anchoring.mlflow, "get_active_trace_id", lambda: None, raising=False)
    monkeypatch.setattr(anchoring.mlflow, "get_tracking_uri", lambda: "file:./mlruns")
    # Default no-op artifact_checksums for tests that don't override.
    monkeypatch.setattr(anchoring, "artifact_checksums",
                        lambda run_id, artifact_path="model": {f"{artifact_path}/m.pkl": "deadbeef"})

    return set_tags, rm_set_tags, log_artifacts_calls, upload_calls, _FakeAnchor()


def test_anchor_writes_payload_json_artifact(monkeypatch, tmp_path):
    """anchor() must write the canonical bytes as ario/payload.json so a
    verifier can recompute the payload_hash without depending on this
    plugin's canonicalization function."""
    import ario_mlflow.anchoring as anchoring

    set_tags, _, log_artifacts_calls, _, fake_anchor = _make_anchor_stubs(monkeypatch)

    result = anchoring.anchor(
        proof_engine=anchoring.ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=fake_anchor,
    )

    assert log_artifacts_calls, "anchor() did not log_artifacts"
    files = log_artifacts_calls[0]["files"]
    assert "payload.json" in files, list(files.keys())
    on_disk = files["payload.json"]
    # The bytes on disk must be exactly the canonical bytes that were hashed.
    assert on_disk == result["payload_bytes"]
    # Hashing them reproduces the envelope's payload_hash — this is what
    # check 2 of the verification flow does.
    from ario_mlflow.proof import hash_data
    assert hash_data(on_disk) == result["envelope"]["payload_hash"]
    # Also: the proof.json artifact matches the envelope.
    assert "proof.json" in files
    proof_on_disk = json.loads(files["proof.json"])
    assert proof_on_disk == result["envelope"]


def test_anchor_envelope_is_pure_commitment_no_source_data(monkeypatch, tmp_path):
    """The envelope on Arweave must not carry params, metrics, or
    artifact_checksums — only the hash."""
    import ario_mlflow.anchoring as anchoring

    *_, fake_anchor = _make_anchor_stubs(monkeypatch)

    result = anchoring.anchor(
        proof_engine=anchoring.ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=fake_anchor,
    )

    env = result["envelope"]
    assert env["event_type"] == "training_complete"
    assert env["payload_hash"]
    # Source data must NOT be in the envelope.
    assert "params" not in env
    assert "metrics" not in env
    assert "artifact_checksums" not in env
    assert "record" not in env  # ensure we didn't slip back to v1 shape


def test_anchor_omits_tracking_uri_from_subject_by_default(monkeypatch, tmp_path):
    """Privacy default: subject does not leak the tracking URI."""
    import ario_mlflow.anchoring as anchoring

    *_, fake_anchor = _make_anchor_stubs(monkeypatch)

    result = anchoring.anchor(
        proof_engine=anchoring.ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=fake_anchor,
    )

    assert result["envelope"]["subject"] == {"type": "mlflow_run", "run_id": "run-test"}
    assert "tracking_uri" not in result["envelope"]["subject"]
    assert "mlflow_tracking_uri" not in result["payload"]


def test_anchor_includes_tracking_uri_when_caller_opts_in(monkeypatch, tmp_path):
    """Caller explicitly opts in via metadata={'include_tracking_uri': True}."""
    import ario_mlflow.anchoring as anchoring

    *_, fake_anchor = _make_anchor_stubs(monkeypatch)

    result = anchoring.anchor(
        proof_engine=anchoring.ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=fake_anchor,
        metadata={"include_tracking_uri": True},
    )

    assert result["envelope"]["subject"]["tracking_uri"] == "file:./mlruns"
    assert result["payload"]["mlflow_tracking_uri"] == "file:./mlruns"


def test_anchor_merges_caller_metadata_into_payload(monkeypatch, tmp_path):
    """OTel correlation, service_name, etc. flow through metadata into the
    canonical payload (and therefore into the signed commitment)."""
    import ario_mlflow.anchoring as anchoring

    *_, fake_anchor = _make_anchor_stubs(monkeypatch)

    result = anchoring.anchor(
        proof_engine=anchoring.ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=fake_anchor,
        metadata={"otel_trace_id": "abc123", "service_name": "test-app"},
    )

    assert result["payload"]["otel_trace_id"] == "abc123"
    assert result["payload"]["service_name"] == "test-app"


def test_anchor_metadata_cannot_overwrite_structural_fields(monkeypatch, tmp_path):
    """Caller metadata must not silently mutate event_type, run_id, params."""
    import ario_mlflow.anchoring as anchoring

    *_, fake_anchor = _make_anchor_stubs(monkeypatch)

    result = anchoring.anchor(
        proof_engine=anchoring.ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=fake_anchor,
        metadata={"event_type": "ATTACKER", "run_id": "FAKE", "params": {"x": "y"}},
    )

    assert result["payload"]["event_type"] == "training_complete"
    assert result["payload"]["run_id"] == "run-test"
    assert result["payload"]["params"] == {}


def test_anchor_chains_to_existing_last_training_hash(monkeypatch, tmp_path):
    """When a registered model exists with ario.last_training_hash, the new
    proof's previous_hash equals that tag."""
    import ario_mlflow.anchoring as anchoring

    set_tags, rm_set_tags, _, upload_calls, fake_anchor = _make_anchor_stubs(
        monkeypatch,
        registered_models_for_run=["fraud-detector"],
        rm_tags={"ario.last_training_hash": "PRIOR-PAYLOAD-HASH"},
        anchor_enabled=True,
        upload_returns={"tx_id": "TX-NEW", "url": "https://example/TX-NEW", "receipt": {}},
    )

    result = anchoring.anchor(
        proof_engine=anchoring.ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=fake_anchor,
    )

    assert result["previous_hash"] == "PRIOR-PAYLOAD-HASH"
    assert result["envelope"]["previous_hash"] == "PRIOR-PAYLOAD-HASH"
    assert result["registered_model"] == "fraud-detector"
    # Chain head was updated to the new payload_hash after successful upload.
    assert rm_set_tags["fraud-detector"]["ario.last_training_hash"] == result["payload_hash"]


def test_anchor_starts_chain_at_genesis_when_no_prior_training(monkeypatch, tmp_path):
    """First training of a model: previous_hash is GENESIS, chain head
    written for next time."""
    import ario_mlflow.anchoring as anchoring

    _, rm_set_tags, _, _, fake_anchor = _make_anchor_stubs(
        monkeypatch,
        registered_models_for_run=["fresh-model"],
        rm_tags={},  # no prior chain head
        anchor_enabled=True,
        upload_returns={"tx_id": "TX-1", "url": "https://example/TX-1", "receipt": {}},
    )

    result = anchoring.anchor(
        proof_engine=anchoring.ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=fake_anchor,
    )

    assert result["previous_hash"] == "GENESIS"
    assert rm_set_tags["fresh-model"]["ario.last_training_hash"] == result["payload_hash"]


def test_anchor_skips_chain_head_when_no_registered_model(monkeypatch, tmp_path):
    """If no registered model points to this run yet (e.g. log_model
    without registered_model_name), the chain-head update is skipped —
    the next registration via ArioMlflowClient picks it up."""
    import ario_mlflow.anchoring as anchoring

    _, rm_set_tags, _, _, fake_anchor = _make_anchor_stubs(
        monkeypatch,
        registered_models_for_run=[],
        anchor_enabled=True,
        upload_returns={"tx_id": "TX-X", "url": "https://example/TX-X", "receipt": {}},
    )

    result = anchoring.anchor(
        proof_engine=anchoring.ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=fake_anchor,
    )

    assert result["previous_hash"] == "GENESIS"
    assert result["registered_model"] is None
    assert rm_set_tags == {}


def test_anchor_does_not_update_chain_head_on_upload_failure(monkeypatch, tmp_path):
    """If the upload fails, ario.last_training_hash must not be advanced
    to a payload that isn't on Arweave — that would poison the next
    proof's previous_hash."""
    import ario_mlflow.anchoring as anchoring

    _, rm_set_tags, _, _, fake_anchor = _make_anchor_stubs(
        monkeypatch,
        registered_models_for_run=["model-x"],
        rm_tags={"ario.last_training_hash": "PRIOR"},
        anchor_enabled=True,
        upload_returns=None,  # simulate upload failure
    )

    anchoring.anchor(
        proof_engine=anchoring.ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub")),
        arweave=fake_anchor,
    )

    # No tag write on the registered model — chain head stays at PRIOR
    # for the next proof to chain from.
    assert rm_set_tags == {}


def test_verified_model_resolves_alias_uri(monkeypatch):
    """Regression for CodeRabbit r4 #3: models:/name@alias must use get_model_version_by_alias."""
    import ario_mlflow.model as model_module
    from ario_mlflow.model import VerifiedModel

    class _FakeMV:
        name = "fraud-detector"
        version = 7
        run_id = "run-alias"
        source = "runs:/run-alias/model"

    calls: list[str] = []

    class _FakeClient:
        def get_model_version_by_alias(self, name, alias):
            calls.append(f"by_alias:{name}:{alias}")
            return _FakeMV()

        def get_model_version(self, *a, **kw):
            calls.append("by_version_WRONG")
            raise AssertionError("numeric API was used for an alias URI")

        def get_run(self, run_id):
            return type("R", (), {"data": type("D", (), {"tags": {}})()})()

    monkeypatch.setattr(model_module.mlflow.tracking, "MlflowClient", lambda: _FakeClient())
    monkeypatch.setattr(model_module, "artifact_checksums", lambda *a, **kw: {})
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model", lambda uri: object())

    vm = VerifiedModel("models:/fraud-detector@champion")
    assert calls == ["by_alias:fraud-detector:champion"]
    assert vm.model_name == "fraud-detector"
    assert vm.model_version == "7"
    assert vm.run_id == "run-alias"


def test_verified_model_predict_produces_pure_commitment_envelope(monkeypatch):
    """VerifiedModel.predict() must commit only hashes — no raw input/output
    in the envelope. Privacy-by-construction."""
    import ario_mlflow.model as model_module
    from ario_mlflow.model import VerifiedModel

    # No artifact integrity check (mv lookup fails → run_id stays "unknown")
    monkeypatch.setattr(model_module, "_resolve_model_version", lambda c, u: None)
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model",
                        lambda uri: type("M", (), {"predict": lambda self, x: [1]})())
    monkeypatch.setattr(model_module.mlflow.tracking, "MlflowClient", lambda: type("C", (), {})())
    monkeypatch.setattr(model_module.mlflow, "get_active_trace_id", lambda: None, raising=False)

    captured = {}

    class _FakeAnchor:
        enabled = False
        def upload_proof(self, env, *a, **kw): return None

    vm = VerifiedModel("models:/foo/1", anchor=_FakeAnchor())
    result = vm.predict({"f1": 1.0, "f2": 2.0})

    # The recorded "envelope" goes through to anchor — capture via the
    # canonical payload that's exposed on the result.record (we kept this
    # name for VerifiedPrediction back-compat).
    payload = result.record
    # Only hashes, no raw values
    assert "input_hash" in payload
    assert "output_hash" in payload
    assert payload["event_type"] == "prediction"
    assert "input" not in payload
    assert "output" not in payload
    assert "prediction" not in payload  # raw prediction must not be in canonical bytes


def test_verified_model_predict_chains_to_registration_tx(monkeypatch):
    """When mv has ario.registration_tx, predictions chain to it.

    Verified by inspecting the canonical-payload internals via the ProofEngine
    monkeypatch — we capture the previous_hash that VerifiedModel passes in.
    """
    import ario_mlflow.model as model_module
    from ario_mlflow.model import VerifiedModel

    captured = {"create_commitment_calls": []}

    class _FakeMV:
        name = "fraud-detector"
        version = 3
        run_id = "r"
        source = "runs:/r/model"
        tags = {"ario.registration_tx": "TX-REG-42"}

    class _FakeRun:
        data = type("D", (), {"tags": {}})()

    class _FakeClient:
        def get_run(self, rid): return _FakeRun()

    monkeypatch.setattr(model_module, "_resolve_model_version", lambda c, u: _FakeMV())
    monkeypatch.setattr(model_module.mlflow.tracking, "MlflowClient", lambda: _FakeClient())
    monkeypatch.setattr(model_module, "artifact_checksums", lambda *a, **kw: {})
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model",
                        lambda uri: type("M", (), {"predict": lambda self, x: [1]})())
    monkeypatch.setattr(model_module.mlflow, "get_active_trace_id", lambda: None, raising=False)

    class _FakeProofEngine:
        def create_commitment(self, *, event_type, subject, payload_bytes, previous_hash, **_):
            captured["create_commitment_calls"].append({
                "event_type": event_type,
                "subject": subject,
                "payload_bytes": payload_bytes,
                "previous_hash": previous_hash,
            })
            return {
                "event_type": event_type, "subject": subject,
                "payload_hash": "PH", "previous_hash": previous_hash,
                "public_key": "PK", "signature": "S",
            }

    class _FakeAnchor:
        enabled = False
        def upload_proof(self, *a, **kw): return None

    vm = VerifiedModel("models:/fraud-detector/3",
                      proof_engine=_FakeProofEngine(),
                      anchor=_FakeAnchor())
    vm.predict([1.0, 2.0])

    assert captured["create_commitment_calls"], "no commitment minted"
    last = captured["create_commitment_calls"][-1]
    assert last["previous_hash"] == "TX-REG-42"
    assert last["event_type"] == "prediction"


def test_verified_model_predict_chains_to_genesis_when_no_registration_tx(monkeypatch):
    """Without ario.registration_tx on the model version, predictions chain
    at GENESIS for this VerifiedModel instance."""
    import ario_mlflow.model as model_module
    from ario_mlflow.model import VerifiedModel

    captured = {"create_commitment_calls": []}

    class _FakeMV:
        name = "fresh"
        version = 1
        run_id = "r"
        source = "runs:/r/model"
        tags = {}  # no registration_tx

    class _FakeRun:
        data = type("D", (), {"tags": {}})()

    monkeypatch.setattr(model_module, "_resolve_model_version", lambda c, u: _FakeMV())
    monkeypatch.setattr(model_module.mlflow.tracking, "MlflowClient",
                        lambda: type("C", (), {"get_run": lambda self, rid: _FakeRun()})())
    monkeypatch.setattr(model_module, "artifact_checksums", lambda *a, **kw: {})
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model",
                        lambda uri: type("M", (), {"predict": lambda self, x: [0]})())
    monkeypatch.setattr(model_module.mlflow, "get_active_trace_id", lambda: None, raising=False)

    class _FakeProofEngine:
        def create_commitment(self, *, previous_hash, **kw):
            captured["create_commitment_calls"].append({"previous_hash": previous_hash})
            return {"public_key": "PK", "signature": "S", "payload_hash": "PH",
                    "event_type": kw["event_type"], "subject": kw["subject"],
                    "previous_hash": previous_hash}

    class _FakeAnchor:
        enabled = False
        def upload_proof(self, *a, **kw): return None

    vm = VerifiedModel("models:/fresh/1",
                      proof_engine=_FakeProofEngine(),
                      anchor=_FakeAnchor())
    vm.predict([1.0])

    assert captured["create_commitment_calls"][-1]["previous_hash"] == "GENESIS"


def test_verified_model_predict_writes_payload_artifact_to_source_run(monkeypatch):
    """Per Phase 1.14: predictions write canonical bytes as
    ario/predictions/<decision_id>/payload.json on the model's source run.

    This is the per-prediction equivalent of training's payload.json,
    giving check 2 (anchored bytes intact) something to compare against.
    Trace tags exist for observability but are not authoritative.
    """
    import ario_mlflow.model as model_module
    from ario_mlflow.model import VerifiedModel

    artifacts_logged: list = []

    class _FakeMV:
        name = "m"
        version = 1
        run_id = "source-run-xyz"
        source = "runs:/source-run-xyz/model"
        tags = {"ario.registration_tx": "TX-REG"}

    class _FakeRun:
        data = type("D", (), {"tags": {}})()

    monkeypatch.setattr(model_module, "_resolve_model_version", lambda c, u: _FakeMV())
    monkeypatch.setattr(model_module.mlflow.tracking, "MlflowClient",
                        lambda: type("C", (), {"get_run": lambda self, rid: _FakeRun()})())
    monkeypatch.setattr(model_module, "artifact_checksums", lambda *a, **kw: {})
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model",
                        lambda uri: type("M", (), {"predict": lambda self, x: [0]})())
    monkeypatch.setattr(model_module.mlflow, "get_active_trace_id", lambda: None, raising=False)

    def _capture_artifacts(local_dir, artifact_path, run_id=None):
        snapshot: dict[str, bytes] = {}
        for root, _dirs, files in os.walk(local_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, local_dir)
                with open(fpath, "rb") as f:
                    snapshot[rel] = f.read()
        artifacts_logged.append({
            "run_id": run_id, "artifact_path": artifact_path, "files": snapshot,
        })

    monkeypatch.setattr(model_module.mlflow, "log_artifacts", _capture_artifacts)

    class _FakeAnchor:
        enabled = False
        def upload_proof(self, *a, **kw): return None

    vm = VerifiedModel("models:/m/1", anchor=_FakeAnchor())
    result = vm.predict([1.0, 2.0])

    # Artifact write happened to the source run, not a new run.
    assert artifacts_logged, "VerifiedModel did not log_artifacts"
    call = artifacts_logged[0]
    assert call["run_id"] == "source-run-xyz"
    assert call["artifact_path"] == "ario"

    # The decision_id-keyed path holds payload.json with the canonical
    # bytes we committed to.
    decision_id = result.decision_id
    rel_path = f"predictions/{decision_id}/payload.json"
    assert rel_path in call["files"], list(call["files"].keys())
    on_disk = call["files"][rel_path]
    # Hashing the on-disk bytes reproduces the envelope's payload_hash —
    # this is exactly what check 2 will do at verify time.
    from ario_mlflow.proof import canonical_json
    assert hash_data(on_disk) == hash_data(canonical_json(result.record))


def test_verified_model_predict_subject_carries_run_id_and_decision_id(monkeypatch):
    """Subject must contain enough for the verifier to find the artifact.

    The new mlflow_prediction subject type includes model_run_id (where
    payload.json lives) and decision_id (the path within ario/predictions/).
    """
    import ario_mlflow.model as model_module
    from ario_mlflow.model import VerifiedModel

    captured = {"subject": None}

    class _FakeMV:
        name = "m"
        version = 1
        run_id = "source-run-42"
        source = "runs:/source-run-42/model"
        tags = {}

    class _FakeRun:
        data = type("D", (), {"tags": {}})()

    monkeypatch.setattr(model_module, "_resolve_model_version", lambda c, u: _FakeMV())
    monkeypatch.setattr(model_module.mlflow.tracking, "MlflowClient",
                        lambda: type("C", (), {"get_run": lambda self, rid: _FakeRun()})())
    monkeypatch.setattr(model_module, "artifact_checksums", lambda *a, **kw: {})
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model",
                        lambda uri: type("M", (), {"predict": lambda self, x: [0]})())
    monkeypatch.setattr(model_module.mlflow, "get_active_trace_id", lambda: None, raising=False)
    monkeypatch.setattr(model_module.mlflow, "log_artifacts", lambda *a, **kw: None)

    class _FakeProofEngine:
        def create_commitment(self, *, subject, **kw):
            captured["subject"] = subject
            return {
                "event_type": kw["event_type"], "subject": subject,
                "payload_hash": "PH", "previous_hash": kw["previous_hash"],
                "public_key": "PK", "signature": "S",
            }

    class _FakeAnchor:
        enabled = False
        def upload_proof(self, *a, **kw): return None

    vm = VerifiedModel("models:/m/1",
                      proof_engine=_FakeProofEngine(),
                      anchor=_FakeAnchor())
    result = vm.predict([1.0])

    subject = captured["subject"]
    assert subject["type"] == "mlflow_prediction"
    assert subject["decision_id"] == result.decision_id
    assert subject["model_run_id"] == "source-run-42"


def test_verified_model_predict_passes_metadata_into_payload(monkeypatch):
    """OTel correlation, service_name, etc. flow through metadata into the
    canonical payload."""
    import ario_mlflow.model as model_module
    from ario_mlflow.model import VerifiedModel

    captured = {"payload_bytes": None}

    class _FakeMV:
        name = "m"
        version = 1
        run_id = "r"
        source = "runs:/r/model"
        tags = {}

    class _FakeRun:
        data = type("D", (), {"tags": {}})()

    monkeypatch.setattr(model_module, "_resolve_model_version", lambda c, u: _FakeMV())
    monkeypatch.setattr(model_module.mlflow.tracking, "MlflowClient",
                        lambda: type("C", (), {"get_run": lambda self, rid: _FakeRun()})())
    monkeypatch.setattr(model_module, "artifact_checksums", lambda *a, **kw: {})
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model",
                        lambda uri: type("M", (), {"predict": lambda self, x: [0]})())
    monkeypatch.setattr(model_module.mlflow, "get_active_trace_id", lambda: None, raising=False)

    class _FakeProofEngine:
        def create_commitment(self, *, payload_bytes, **kw):
            captured["payload_bytes"] = payload_bytes
            return {"public_key": "PK", "signature": "S", "payload_hash": "PH",
                    "event_type": kw["event_type"], "subject": kw["subject"],
                    "previous_hash": kw["previous_hash"]}

    class _FakeAnchor:
        enabled = False
        def upload_proof(self, *a, **kw): return None

    vm = VerifiedModel("models:/m/1",
                      proof_engine=_FakeProofEngine(),
                      anchor=_FakeAnchor())
    vm.predict([1.0], metadata={"otel_trace_id": "ot", "service_name": "svc"})

    payload = json.loads(captured["payload_bytes"])
    assert payload["otel_trace_id"] == "ot"
    assert payload["service_name"] == "svc"
    # Structural fields not overwritten
    assert payload["event_type"] == "prediction"
    assert "input_hash" in payload


def test_verified_model_resolves_stage_uri(monkeypatch):
    """Regression for CodeRabbit r4 #3: models:/name/Production uses search_model_versions."""
    import ario_mlflow.model as model_module
    from ario_mlflow.model import VerifiedModel

    class _FakeMV:
        name = "fraud-detector"
        version = 3
        run_id = "run-stage"
        source = "runs:/run-stage/model"

    calls: list[str] = []

    class _FakeClient:
        def search_model_versions(self, query):
            calls.append(f"search:{query}")
            return [_FakeMV()]

        def get_model_version(self, *a, **kw):
            calls.append("by_version_WRONG")
            raise AssertionError("numeric API was used for a stage URI")

        def get_run(self, run_id):
            return type("R", (), {"data": type("D", (), {"tags": {}})()})()

    monkeypatch.setattr(model_module.mlflow.tracking, "MlflowClient", lambda: _FakeClient())
    monkeypatch.setattr(model_module, "artifact_checksums", lambda *a, **kw: {})
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model", lambda uri: object())

    vm = VerifiedModel("models:/fraud-detector/Production")
    assert len(calls) == 1 and "Production" in calls[0]
    assert vm.model_name == "fraud-detector"
    assert vm.model_version == "3"


# --- Previously missed out-of-diff regressions ----------------------------


def test_ario_client_uses_source_uri_for_run_id_fallback(monkeypatch):
    """When create_model_version is called with run_id=None but source is a runs:/
    URI, the registration must still link to training (not mint a GENESIS proof).

    Adapted for the pure-commitment redesign: assertions now target
    create_commitment instead of the legacy create_proof, but the
    underlying contract — registration proofs chain to ario.training_tx
    on the source run — is unchanged.
    """
    import ario_mlflow.client as client_module

    captured: dict = {"commitment_calls": [], "upload_calls": 0}

    class _FakeRun:
        data = type("D", (), {"tags": {
            "ario.training_tx": "TX-training-123",
            "ario.artifact_hash": "expected-hash",
        }})()

    class _Client(client_module.ArioMlflowClient):
        def __init__(self):
            def _capture_commitment(*, event_type, subject, payload_bytes, previous_hash, **_):
                captured["commitment_calls"].append({
                    "event_type": event_type,
                    "subject": subject,
                    "payload_bytes": payload_bytes,
                    "previous_hash": previous_hash,
                })
                return {
                    "event_type": event_type,
                    "subject": subject,
                    "payload_hash": "H",
                    "previous_hash": previous_hash,
                    "public_key": "PK",
                    "signature": "SIG",
                }

            self._proof_engine = type(
                "PE",
                (),
                {"create_commitment": lambda self_, **kw: _capture_commitment(**kw)},
            )()
            self._anchor = type(
                "A",
                (),
                {
                    "enabled": False,
                    "upload_proof": lambda self_, *a, **kw: (
                        captured.__setitem__("upload_calls", captured["upload_calls"] + 1) or None
                    ),
                },
            )()

        def get_run(self, run_id):
            captured["get_run_called_with"] = run_id
            return _FakeRun()

        def set_model_version_tag(self, *a, **kw): pass
        def log_artifacts(self, *a, **kw): pass

    monkeypatch.setattr(client_module, "artifact_checksums", lambda *a, **kw: {})

    c = _Client()
    c._anchor_registration("fraud", "1", run_id=None, source="runs:/train-abc/sklearn-model")

    # Key assertions:
    # - get_run was called with the run_id parsed from source
    assert captured.get("get_run_called_with") == "train-abc", captured
    # - The registration commitment chains to the training_tx (not GENESIS)
    assert captured["commitment_calls"], "No registration commitment was minted"
    last = captured["commitment_calls"][-1]
    assert last["previous_hash"] == "TX-training-123", (
        f"Expected registration commitment to chain to training_tx; "
        f"got previous_hash={last['previous_hash']!r}"
    )
    # Subject identifies the model version, not the run
    assert last["subject"] == {"type": "mlflow_model_version", "name": "fraud", "version": "1"}
    # The canonical bytes carry source_run_id (parsed from source URI)
    payload = json.loads(last["payload_bytes"])
    assert payload["source_run_id"] == "train-abc"
    assert payload["event_type"] == "model_registered"


def test_ario_client_registration_writes_payload_json_artifact(monkeypatch):
    """Registration must persist the canonical bytes as ario/registration_payload.json
    so verifiers can recompute the payload_hash without depending on this plugin."""
    import ario_mlflow.client as client_module

    artifacts_logged: list = []

    class _FakeRun:
        data = type("D", (), {"tags": {
            "ario.training_tx": "TX-train",
            "ario.artifact_hash": "h",
        }})()

    class _Client(client_module.ArioMlflowClient):
        def __init__(self):
            from ario_mlflow.proof import ProofEngine
            import tempfile as _t
            self._proof_engine = ProofEngine(
                str(_t.mkdtemp() + "/priv"),
                str(_t.mkdtemp() + "/pub"),
            )
            self._anchor = type("A", (), {"enabled": False, "upload_proof": lambda *a, **k: None})()

        def get_run(self, run_id):
            return _FakeRun()

        def set_model_version_tag(self, *a, **kw): pass

        def log_artifacts(self, source_run_id, local_dir, ap):
            snapshot: dict[str, bytes] = {}
            for root, _dirs, files in os.walk(local_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    rel = os.path.relpath(fpath, local_dir)
                    with open(fpath, "rb") as f:
                        snapshot[rel] = f.read()
            artifacts_logged.append({"run_id": source_run_id, "files": snapshot})

    monkeypatch.setattr(client_module, "artifact_checksums", lambda *a, **kw: {})

    c = _Client()
    c._anchor_registration("fraud", "2", run_id="train-xyz", source="runs:/train-xyz/model")

    assert artifacts_logged, "log_artifacts was not called"
    files = artifacts_logged[0]["files"]
    assert "registration_payload.json" in files, list(files.keys())
    assert "registration_proof.json" in files
    # Hashing the saved payload reproduces the envelope's payload_hash.
    proof = json.loads(files["registration_proof.json"])
    assert hash_data(files["registration_payload.json"]) == proof["payload_hash"]


def test_ario_client_promotion_chains_to_registration_tx(monkeypatch):
    """Promotion proofs must chain to ario.registration_tx on the model version."""
    import ario_mlflow.client as client_module

    captured: dict = {"commitment_calls": []}

    class _FakeMV:
        tags = {"ario.registration_tx": "TX-REG-99"}

    class _Client(client_module.ArioMlflowClient):
        def __init__(self):
            def _capture(*, event_type, subject, payload_bytes, previous_hash, **_):
                captured["commitment_calls"].append({
                    "event_type": event_type,
                    "subject": subject,
                    "payload_bytes": payload_bytes,
                    "previous_hash": previous_hash,
                })
                return {
                    "event_type": event_type, "subject": subject,
                    "payload_hash": "PH", "previous_hash": previous_hash,
                    "public_key": "PK", "signature": "S",
                }

            self._proof_engine = type("PE", (), {
                "create_commitment": lambda self_, **kw: _capture(**kw),
            })()
            self._anchor = type("A", (), {"enabled": False, "upload_proof": lambda *a, **k: None})()

        def get_model_version(self, name, version): return _FakeMV()
        def set_model_version_tag(self, *a, **kw): pass

    c = _Client()
    c._anchor_promotion("fraud", "5", "Staging", "Production")

    assert captured["commitment_calls"], "no promotion commitment minted"
    last = captured["commitment_calls"][-1]
    assert last["event_type"] == "stage_transition"
    assert last["previous_hash"] == "TX-REG-99"
    payload = json.loads(last["payload_bytes"])
    assert payload["from_stage"] == "Staging"
    assert payload["to_stage"] == "Production"


def test_ario_client_registration_passes_metadata_through(monkeypatch):
    """create_model_version's metadata kwarg must reach the canonical payload."""
    import ario_mlflow.client as client_module

    captured: dict = {"commitment_calls": []}

    class _FakeRun:
        data = type("D", (), {"tags": {"ario.training_tx": "TX-T", "ario.artifact_hash": "h"}})()

    class _Client(client_module.ArioMlflowClient):
        def __init__(self):
            def _capture(*, event_type, subject, payload_bytes, previous_hash, **_):
                captured["commitment_calls"].append({"payload_bytes": payload_bytes})
                return {
                    "event_type": event_type, "subject": subject,
                    "payload_hash": "PH", "previous_hash": previous_hash,
                    "public_key": "PK", "signature": "S",
                }
            self._proof_engine = type("PE", (), {
                "create_commitment": lambda self_, **kw: _capture(**kw),
            })()
            self._anchor = type("A", (), {"enabled": False, "upload_proof": lambda *a, **k: None})()

        def get_run(self, rid): return _FakeRun()
        def set_model_version_tag(self, *a, **kw): pass
        def log_artifacts(self, *a, **kw): pass

    monkeypatch.setattr(client_module, "artifact_checksums", lambda *a, **kw: {})

    c = _Client()
    c._anchor_registration(
        "m", "1", run_id="r", source="runs:/r/model",
        metadata={"otel_trace_id": "ot-1", "service_name": "s"},
    )

    payload = json.loads(captured["commitment_calls"][-1]["payload_bytes"])
    assert payload["otel_trace_id"] == "ot-1"
    assert payload["service_name"] == "s"
    # Structural fields not overwritten
    assert payload["event_type"] == "model_registered"
    assert payload["model_name"] == "m"


def test_arweave_upload_proof_sets_post_timeout():
    """POST must not hang indefinitely — matches the sibling GET's 30s bound."""
    import ario_mlflow.arweave as arweave_module

    captured: dict = {}

    class _FakeResp:
        status_code = 200
        def json(self): return {"id": "TX123"}

    def _fake_post(url, *, data=None, headers=None, timeout=None):
        captured["timeout"] = timeout
        return _FakeResp()

    # Construct an anchor with the internals populated enough to reach the POST.
    anchor = arweave_module.ArweaveAnchor.__new__(arweave_module.ArweaveAnchor)
    anchor.enabled = True
    anchor.gateway_host = "turbo-gateway.com"
    anchor.gateways = ["turbo-gateway.com"]
    anchor.last_error = None

    class _FakeSigner:
        def get_wallet_address(self): return "addr"

    anchor._signer = _FakeSigner()
    anchor._upload_url = "https://upload.example"
    anchor._token = "token"

    # Capture the post call via the session's post attribute — the real
    # session would route through urllib3's retry adapter, but for this
    # test we only care that upload_proof passed an explicit timeout.
    fake_session = MagicMock()
    fake_session.post = _fake_post
    anchor._session = fake_session

    # Stub the data-item creation path so we don't depend on turbo_sdk internals.
    class _FakeDataItem:
        def get_raw(self): return b"payload"

    import sys, types
    fake_bundle = types.ModuleType("turbo_sdk.bundle")
    fake_bundle.create_data = lambda data, signer, tags: _FakeDataItem()
    fake_bundle.sign = lambda item, signer: None
    sys.modules["turbo_sdk.bundle"] = fake_bundle

    result = anchor.upload_proof({"record": {"event_type": "x"}, "record_hash": "h"})
    assert result is not None
    # The test's real purpose: the POST carried an explicit timeout.
    assert captured.get("timeout") is not None, "session.post was called without timeout"
    assert captured["timeout"] >= 10, captured["timeout"]


def test_plugin_version_tag_reflects_installed_package():
    """ario.version must come from package metadata, not a hardcoded string."""
    from ario_mlflow.plugin import ArioContextProvider, _plugin_version

    provider = ArioContextProvider()
    tags = provider.tags()
    assert tags["ario.enabled"] == "true"
    # Must match whatever importlib reports (may be the real version or "unknown").
    assert tags["ario.version"] == _plugin_version()
    # Must not be the stale hardcoded string from before this fix.
    # (If the installed version happens to be exactly "0.1.0", the helper returns
    # that same value; we just assert the source of truth is importlib, which is
    # what _plugin_version() exercises.)


def test_verified_prediction_uses_Any_not_lowercase_any():
    """Regression: dataclass annotation must be a real type hint (Any), not the
    lowercase builtin ``any``. Static type checkers treat ``any`` as the builtin
    function and reject it."""
    from ario_mlflow.model import VerifiedPrediction

    hints = VerifiedPrediction.__annotations__
    # The annotation can be stored as the actual Any type or as the string "Any",
    # depending on `from __future__ import annotations`. It must never be the
    # builtin ``any`` function.
    assert hints["prediction"] is not any
    assert hints["prediction"] in (
        __import__("typing").Any,
        "Any",
    ), hints["prediction"]


def test_ario_client_skips_registration_anchor_on_get_run_failure(monkeypatch, caplog):
    """Regression for CodeRabbit r3 #2: don't mint a bad GENESIS proof when the
    source run lookup fails transiently."""
    import ario_mlflow.client as client_module

    captured: dict = {"create_commitment_called": False, "upload_called": False}

    class _FailingClient(client_module.ArioMlflowClient):
        def __init__(self):
            # Skip MlflowClient __init__; we override everything we touch.
            self._proof_engine = type(
                "PE", (), {"create_commitment": lambda *a, **kw: (captured.__setitem__("create_commitment_called", True) or {})}
            )()
            self._anchor = type(
                "A", (), {"enabled": True, "upload_proof": lambda *a, **kw: (captured.__setitem__("upload_called", True) or None)}
            )()

        def get_run(self, run_id):
            raise RuntimeError("tracking store down")

        def set_model_version_tag(self, *a, **kw):
            captured["set_tag_called"] = True

        def log_artifacts(self, *a, **kw):
            captured["log_artifacts_called"] = True

    c = _FailingClient()
    with caplog.at_level("WARNING"):
        c._anchor_registration("fraud", "1", "run-id", "runs:/run-id/model")

    assert captured["create_commitment_called"] is False, (
        "Must not mint a registration proof when source-run lookup failed"
    )
    assert captured["upload_called"] is False
    assert not captured.get("set_tag_called")
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("Skipping registration anchoring" in m for m in warnings), warnings


# Phase 2.D removed the demo's local chain-integrity feature
# (compute_chain_integrity + /api/chain-integrity endpoint + the
# chain-status widget). The on-chain DAG via Arweave tags is the
# canonical chain; per-event verify is done via the plugin's
# full_verify. Tests for the removed local helper deleted with it.


# --- prediction check 3 (live MLflow re-derivation via trace tag) ----------


def test_verify_source_of_truth_for_prediction_passes_when_trace_tag_matches():
    """Phase 2: predictions get check 3 by mirroring the canonical
    payload onto the trace as ``ario.payload_json``. When the trace tag
    matches the anchored artifact, source_of_truth.ok is True."""
    from ario_mlflow.proof import canonical_json
    from ario_mlflow.verify import verify_source_of_truth

    payload = {
        "event_type": "prediction",
        "decision_id": "abc-123",
        "model_name": "credit-scorer",
        "model_version": "1",
        "input_hash": "0xinput",
        "output_hash": "0xoutput",
        "latency_ms": 4.2,
        "mlflow_trace_id": "tr-trace-1",
    }
    payload_bytes = canonical_json(payload)

    # The trace mirrors the artifact verbatim — no tampering.
    class _Trace:
        class info:
            tags = {"ario.payload_json": payload_bytes.decode("utf-8")}

    class _StubClient:
        def get_trace(self, trace_id):
            assert trace_id == "tr-trace-1"
            return _Trace()

    envelope = {"event_type": "prediction"}
    out = verify_source_of_truth(envelope, payload_bytes, _StubClient())
    assert out["ok"] is True, out


def test_verify_source_of_truth_for_prediction_fails_when_trace_tag_tampered():
    """Catches the new tamper vector: someone modifies ``ario.payload_json``
    on the MLflow trace without re-uploading the artifact (or vice versa)."""
    from ario_mlflow.proof import canonical_json
    from ario_mlflow.verify import verify_source_of_truth

    payload = {
        "event_type": "prediction",
        "decision_id": "abc-123",
        "model_name": "credit-scorer",
        "model_version": "1",
        "input_hash": "0xoriginal",
        "output_hash": "0xoutput",
        "latency_ms": 4.2,
        "mlflow_trace_id": "tr-trace-2",
    }
    payload_bytes = canonical_json(payload)

    # Trace tag has been tampered: input_hash differs from artifact.
    tampered = dict(payload)
    tampered["input_hash"] = "0xTAMPERED"
    tampered_json = canonical_json(tampered).decode("utf-8")

    class _Trace:
        class info:
            tags = {"ario.payload_json": tampered_json}

    class _StubClient:
        def get_trace(self, trace_id):
            return _Trace()

    envelope = {"event_type": "prediction"}
    out = verify_source_of_truth(envelope, payload_bytes, _StubClient())
    assert out["ok"] is False, out


def test_verify_source_of_truth_for_prediction_fails_when_trace_pruned():
    """Pruned trace surfaces as a clear failure rather than a silent
    pass — auditors see ``live_refetch_incomplete`` and can act on it."""
    from ario_mlflow.proof import canonical_json
    from ario_mlflow.verify import verify_source_of_truth

    payload = {
        "event_type": "prediction",
        "decision_id": "abc-123",
        "input_hash": "0xa",
        "output_hash": "0xb",
        "mlflow_trace_id": "tr-pruned",
    }
    payload_bytes = canonical_json(payload)

    class _StubClient:
        def get_trace(self, trace_id):
            raise RuntimeError("trace not found (pruned)")

    envelope = {"event_type": "prediction"}
    out = verify_source_of_truth(envelope, payload_bytes, _StubClient())
    assert out["ok"] is False, out
    assert out.get("reason") == "live_refetch_incomplete", out


def test_verify_source_of_truth_for_prediction_fails_without_trace_id():
    """Old-style prediction payloads without ``mlflow_trace_id`` can't be
    re-derived — surfaces as a clear failure with a descriptive reason."""
    from ario_mlflow.proof import canonical_json
    from ario_mlflow.verify import verify_source_of_truth

    payload = {
        "event_type": "prediction",
        "decision_id": "abc-123",
        "input_hash": "0xa",
        "output_hash": "0xb",
        # no mlflow_trace_id
    }
    payload_bytes = canonical_json(payload)

    class _StubClient:
        def get_trace(self, trace_id):
            raise AssertionError("should never be called")

    envelope = {"event_type": "prediction"}
    out = verify_source_of_truth(envelope, payload_bytes, _StubClient())
    assert out["ok"] is False, out
    assert "mlflow_trace_id" in out.get("detail", ""), out


def test_verify_source_of_truth_rejects_non_object_trace_tag():
    """If ario.payload_json on the trace is valid JSON but not an object
    (e.g., a tampered string, list, or number), the refetcher must fail
    closed via LiveRefetchError instead of letting the downstream rebuild
    crash on .update() / .keys()."""
    from ario_mlflow.proof import canonical_json
    from ario_mlflow.verify import verify_source_of_truth

    payload = {
        "event_type": "prediction",
        "decision_id": "abc-123",
        "input_hash": "0xa",
        "output_hash": "0xb",
        "mlflow_trace_id": "tr-bad-tag",
    }
    payload_bytes = canonical_json(payload)

    # Trace tag has been tampered to a JSON string instead of an object.
    class _Trace:
        class info:
            tags = {"ario.payload_json": '"a string instead of an object"'}

    class _StubClient:
        def get_trace(self, trace_id):
            return _Trace()

    envelope = {"event_type": "prediction"}
    out = verify_source_of_truth(envelope, payload_bytes, _StubClient())
    assert out["ok"] is False, out
    assert out.get("reason") == "live_refetch_incomplete", out
    assert "non-object" in out.get("detail", "").lower(), out
