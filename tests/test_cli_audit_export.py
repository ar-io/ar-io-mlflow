"""Tests for `ar-io-mlflow audit --format=json` — the machine-readable
model-lineage evidence bundle (parity with ariod's `audit export`).

Mocks the MLflow client + ar.io components so no server / network is
needed (same discipline as the rest of the suite).
"""

import json

import pytest

from ario_mlflow import cli
from ario_mlflow.proof import ProofEngine, canonical_json


class _StubAnchor:
    """Returns a fixed signed envelope for any tx_id."""

    def __init__(self, envelope):
        self._env = envelope

    def fetch_proof(self, tx_id):
        return self._env


class _StubModelVersion:
    def __init__(self, run_id, tags, current_stage="Production"):
        self.run_id = run_id
        self.tags = tags
        self.current_stage = current_stage


class _StubRun:
    def __init__(self, tags):
        self.data = type("D", (), {"tags": tags})()


class _StubMlflowClient:
    def __init__(self, mv, run):
        self._mv = mv
        self._run = run

    def get_model_version(self, name, version):
        return self._mv

    def get_run(self, run_id):
        return self._run


def _args(**kw):
    return type("Args", (), kw)()


@pytest.fixture
def signed_envelope(tmp_path):
    engine = ProofEngine(str(tmp_path / "priv"), str(tmp_path / "pub"))
    canonical = canonical_json({"run_id": "r-1", "params": {}, "metrics": {}})
    env = engine.create_commitment(
        event_type="training_complete",
        subject={"type": "mlflow_run", "run_id": "r-1"},
        payload_bytes=canonical,
        previous_hash="GENESIS",
    )
    return engine, env


def _wire(monkeypatch, engine, env, mv_tags, run_tags, run_id="r-1", stage="Production"):
    mv = _StubModelVersion(run_id=run_id, tags=mv_tags, current_stage=stage)
    run = _StubRun(run_tags)
    monkeypatch.setattr(cli, "_get_components", lambda: (engine, _StubAnchor(env), None))
    monkeypatch.setattr(cli.mlflow.tracking, "MlflowClient", lambda: _StubMlflowClient(mv, run))


def test_audit_json_to_file_has_expected_shape(tmp_path, monkeypatch, signed_envelope):
    engine, env = signed_envelope
    _wire(
        monkeypatch, engine, env,
        mv_tags={
            "ario.registration_tx": "TX_REG",
            "ario.promotion_tx": "TX_PROMO",
        },
        run_tags={
            "ario.training_tx": "TX_TRAIN",
            "ario.artifact_hash": "a" * 64,
        },
    )
    out_path = tmp_path / "bundle.json"
    rc = cli.cmd_audit(_args(model="fraud-detector/3", format="json", output=str(out_path)))

    assert out_path.exists()
    bundle = json.loads(out_path.read_text())
    assert bundle["schema"] == "ario.mlflow.audit/v1"
    assert bundle["model"] == "fraud-detector"
    assert bundle["version"] == "3"
    assert bundle["artifact_hash"] == "a" * 64
    assert bundle["generated_at"].endswith("Z")
    # Three lineage stages, all anchored in this fixture.
    stage_names = [s["stage"] for s in bundle["stages"]]
    assert stage_names == ["training", "registration", "promotion"]
    for s in bundle["stages"]:
        assert s["anchored"] is True
        assert s["tx_id"] in {"TX_TRAIN", "TX_REG", "TX_PROMO"}
        assert set(s["checks"]) == {"signature", "anchored_bytes", "source_of_truth", "ario_attestation"}
    assert "overall_ok" in bundle
    # rc mirrors overall_ok
    assert rc == (0 if bundle["overall_ok"] else 1)


def test_audit_json_unanchored_stage_renders_null_checks(tmp_path, monkeypatch, signed_envelope):
    engine, env = signed_envelope
    # No registration/promotion tx → those stages are not anchored.
    _wire(
        monkeypatch, engine, env,
        mv_tags={},
        run_tags={"ario.training_tx": "TX_TRAIN"},
    )
    out_path = tmp_path / "bundle.json"
    cli.cmd_audit(_args(model="m/1", format="json", output=str(out_path)))
    bundle = json.loads(out_path.read_text())

    by_stage = {s["stage"]: s for s in bundle["stages"]}
    assert by_stage["registration"]["anchored"] is False
    assert by_stage["registration"]["checks"] is None
    assert by_stage["registration"]["ok"] is None
    # Unanchored stages don't flip overall_ok to False on their own —
    # only a FAILED verification does. With training anchored + a valid
    # envelope, overall stays truthy unless a check failed.
    assert "overall_ok" in bundle


def test_audit_json_to_stdout(tmp_path, monkeypatch, signed_envelope, capsys):
    engine, env = signed_envelope
    _wire(monkeypatch, engine, env, mv_tags={"ario.registration_tx": "TX_REG"}, run_tags={})
    cli.cmd_audit(_args(model="m/1", format="json", output=None))
    captured = capsys.readouterr()
    # stdout must be pipe-clean JSON in json mode (no terminal panel).
    bundle = json.loads(captured.out)
    assert bundle["schema"] == "ario.mlflow.audit/v1"
    # No "Auditing model lineage" banner leaked into stdout.
    assert "Auditing model lineage" not in captured.out


def test_audit_text_mode_still_prints_panel(monkeypatch, signed_envelope, capsys):
    engine, env = signed_envelope
    _wire(monkeypatch, engine, env, mv_tags={"ario.registration_tx": "TX_REG"}, run_tags={})
    cli.cmd_audit(_args(model="m/1", format="text", output=None))
    captured = capsys.readouterr()
    # Text mode unchanged: banner + Overall line present, no JSON.
    assert "Auditing model lineage" in captured.out
    assert "Overall:" in captured.out
    assert '"schema"' not in captured.out


def test_audit_parser_wires_format_and_output():
    parser = cli.build_parser()
    args = parser.parse_args(["audit", "m/1", "--format", "json", "--output", "/tmp/x.json"])
    assert args.command == "audit"
    assert args.format == "json"
    assert args.output == "/tmp/x.json"
    # Defaults: text format, no output file.
    args2 = parser.parse_args(["audit", "m/1"])
    assert args2.format == "text"
    assert args2.output is None
