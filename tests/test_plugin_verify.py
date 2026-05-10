"""Tests for the auditor-shaped ``verify_record`` primitive and the
``verify_proof_by_tx`` operator-side wrapper.

``verify_record`` is the foundation primitive: given an envelope plus
canonical bytes, run signature + payload-hash match + optional ar.io
attestation. No MLflow access required — this is what an auditor uses
against a portable bundle.

``verify_proof_by_tx`` wraps the operator-side flow: pull the envelope
from Arweave, pull canonical bytes from MLflow, then run all four
checks. Adds a ``proof_found`` flag so the demo's "Proof Found" UI row
can distinguish "envelope retrieved" from "envelope was missing."
"""

from __future__ import annotations

from ario_mlflow.proof import ProofEngine, canonical_json


# --- verify_record (foundation primitive, auditor-shaped) -----------------


def test_verify_record_passes_with_valid_envelope_and_matching_canonical_bytes(tmp_path):
    """Auditor's happy path. Envelope plus canonical bytes that hash to
    its ``payload_hash``, signature valid → both sub-checks pass and
    overall is True. ar.io is skipped (no client) — neutral, not a
    failure."""
    from ario_mlflow.verify import verify_record

    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    canonical = canonical_json({"params": {"lr": 0.01}, "metrics": {"acc": 0.91}})
    env = engine.create_commitment(
        event_type="training_complete",
        subject={"type": "mlflow_run", "run_id": "r-1"},
        payload_bytes=canonical,
        previous_hash="GENESIS",
    )

    out = verify_record(env, canonical, proof_engine=engine)

    assert out["signature"]["ok"] is True
    assert out["anchored_bytes"]["ok"] is True
    assert out["ario_attestation"]["ok"] is None
    assert out["overall"] is True


def test_verify_record_fails_when_canonical_bytes_dont_match_payload_hash(tmp_path):
    """If the bytes the auditor was handed don't hash to the envelope's
    ``payload_hash``, the anchored-bytes check fails. Signature is still
    valid (the envelope itself wasn't touched) but overall is False —
    the bundle's records don't match what was anchored."""
    from ario_mlflow.verify import verify_record

    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    canonical = canonical_json({"original": "payload"})
    env = engine.create_commitment(
        event_type="training_complete",
        subject={"type": "mlflow_run", "run_id": "r-1"},
        payload_bytes=canonical,
        previous_hash="GENESIS",
    )

    out = verify_record(env, b'{"tampered":"payload"}', proof_engine=engine)

    assert out["signature"]["ok"] is True
    assert out["anchored_bytes"]["ok"] is False
    assert out["overall"] is False


def test_verify_record_skips_attestation_when_ario_client_is_none(tmp_path):
    """No ar.io client → ``ario_attestation.ok`` is None ('not checked'),
    not False. A bundle verified offline still gets a True overall when
    sig + bytes match."""
    from ario_mlflow.verify import verify_record

    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    canonical = canonical_json({"k": "v"})
    env = engine.create_commitment(
        event_type="prediction",
        subject={
            "type": "mlflow_prediction",
            "decision_id": "d",
            "model_run_id": "r",
        },
        payload_bytes=canonical,
        previous_hash="GENESIS",
    )

    out = verify_record(env, canonical, proof_engine=engine, ario_client=None)

    assert out["ario_attestation"]["ok"] is None
    assert out["signature"]["ok"] is True
    assert out["anchored_bytes"]["ok"] is True
    assert out["overall"] is True


# --- verify_proof_by_tx (operator-side wrapper) ---------------------------


def test_verify_proof_by_tx_returns_proof_found_false_when_fetch_fails(tmp_path):
    """When ``ArweaveAnchor.fetch_proof`` returns None (TX missing or
    gateway unreachable), ``proof_found`` is False and every sub-check
    returns ok=None — the checks weren't run."""
    from ario_mlflow.verify import verify_proof_by_tx

    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))

    class _MissingAnchor:
        def fetch_proof(self, tx_id):
            return None

    out = verify_proof_by_tx(
        "TX-missing",
        anchor=_MissingAnchor(),
        proof_engine=engine,
    )

    assert out["proof_found"] is False
    assert out["signature"]["ok"] is None
    assert out["anchored_bytes"]["ok"] is None
    assert out["source_of_truth"]["ok"] is None
    assert out["ario_attestation"]["ok"] is None
    assert out["overall"] is None


def test_verify_proof_by_tx_returns_proof_found_true_with_valid_envelope(tmp_path):
    """Happy operator path on the fetch step alone — fetch succeeds,
    ``proof_found`` is True. Signature can be checked envelope-only;
    bytes/source-of-truth need MLflow (None when omitted in this test)."""
    from ario_mlflow.verify import verify_proof_by_tx

    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    canonical = canonical_json({"k": "v"})
    env = engine.create_commitment(
        event_type="training_complete",
        subject={"type": "mlflow_run", "run_id": "r-1"},
        payload_bytes=canonical,
        previous_hash="GENESIS",
    )

    class _StubAnchor:
        def fetch_proof(self, tx_id):
            return env

    out = verify_proof_by_tx(
        "TX-found",
        anchor=_StubAnchor(),
        proof_engine=engine,
    )

    assert out["proof_found"] is True
    assert out["signature"]["ok"] is True
    # Without mlflow_client, bytes / source-of-truth aren't run.
    assert out["anchored_bytes"]["ok"] is None
    assert out["source_of_truth"]["ok"] is None
    # All four-check keys plus proof_found are present.
    assert {"proof_found", "signature", "anchored_bytes",
            "source_of_truth", "ario_attestation", "overall"}.issubset(out.keys())


def test_verify_proof_by_tx_includes_source_of_truth_check(tmp_path, monkeypatch):
    """``source_of_truth`` is part of the operator-side wrapper (NOT
    ``verify_record``). When a mlflow_client is provided and the
    canonical artifact is present, source-of-truth runs against the
    anchored bytes and passes when the live re-derivation matches."""
    from ario_mlflow.verify import verify_proof_by_tx

    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    payload_dict = {
        "run_id": "r-sot",
        "params": {},
        "metrics": {},
        "artifact_checksums": {},
        "source_name": "",
        "git_commit": "",
    }
    canonical = canonical_json(payload_dict)
    env = engine.create_commitment(
        event_type="training_complete",
        subject={"type": "mlflow_run", "run_id": "r-sot"},
        payload_bytes=canonical,
        previous_hash="GENESIS",
    )

    class _StubAnchor:
        def fetch_proof(self, tx_id):
            return env

    artifact_file = tmp_path / "payload.json"
    artifact_file.write_bytes(canonical)

    import mlflow.artifacts
    monkeypatch.setattr(
        mlflow.artifacts, "download_artifacts",
        lambda **kw: str(artifact_file),
    )

    class _Run:
        class _Data:
            def __init__(self):
                self.params = {}
                self.metrics = {}
                self.tags = {
                    "mlflow.source.name": "",
                    "mlflow.source.git.commit": "",
                }

        def __init__(self):
            self.data = self._Data()

    class _StubMlflowClient:
        def get_run(self, run_id):
            return _Run()

    import ario_mlflow.anchoring as anchoring
    monkeypatch.setattr(
        anchoring, "artifact_checksums",
        lambda run_id, artifact_path="model": {},
    )
    monkeypatch.setattr(anchoring, "_logged_model_paths", lambda run: [])

    out = verify_proof_by_tx(
        "TX-sot",
        anchor=_StubAnchor(),
        proof_engine=engine,
        mlflow_client=_StubMlflowClient(),
    )

    assert out["proof_found"] is True
    assert out["source_of_truth"]["ok"] is True, out["source_of_truth"]
