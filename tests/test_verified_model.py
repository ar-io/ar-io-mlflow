"""Tests for the VerifiedModel agent verify-status gate (v1.2 Lane E).

The gate is exercised with fake VerifyStatusClient objects — no HTTP here
(the client's wire behavior is covered by test_verify_status_client.py).
MLflow surfaces are stubbed with the same monkeypatch pattern as the
existing VerifiedModel tests in test_plugin_smoke.py.
"""

import logging

import pytest

from ario_mlflow.errors import (
    AssetMissingError,
    AssetStaleError,
    AssetTamperedError,
    AssetUnknownError,
    AssetVerificationError,
    VerifyStatusTransportError,
)
from ario_mlflow.verify_status_client import VerifyStatus


def make_status(outcome="verified", stale=False, **overrides):
    fields = dict(
        asset_id="fraud-model",
        tenant_id="acme-prod",
        agent_id="agent-01",
        outcome=outcome,
        last_verified_at="2026-06-10T14:23:11.420Z",
        max_age=3600,
        stale=stale,
        policy_hash="a3f1b2c4",
        current_tx_id="TxAbc123",
    )
    fields.update(overrides)
    return VerifyStatus(**fields)


class FakeVerifyStatusClient:
    """Duck-typed VerifyStatusClient: canned status or canned exception."""

    def __init__(self, status=None, exc=None):
        self._status = status
        self._exc = exc
        self.calls: list[str] = []

    def get(self, asset_id, **kwargs):
        self.calls.append(asset_id)
        if self._exc is not None:
            raise self._exc
        return self._status


def _stub_mlflow_surfaces(monkeypatch, calls):
    """Stub every MLflow surface VerifiedModel touches; record call order."""
    import ario_mlflow.model as model_module
    from ario_mlflow.proof import canonical_json, hash_data

    checksums = {"model/foo": "deadbeef"}
    expected = hash_data(canonical_json(checksums))

    class _FakeRun:
        data = type("D", (), {"tags": {"ario.artifact_hash": expected}})()

    class _FakeMV:
        name = "foo"
        version = 1
        run_id = "run-xyz"
        source = "runs:/run-xyz/model"
        tags = {}

    class _FakeClient:
        def get_model_version(self, name, version):
            return _FakeMV()

        def get_run(self, run_id):
            return _FakeRun()

    def _fake_checksums(run_id, *a, **kw):
        calls.append("integrity")
        return checksums

    def _fake_load_model(uri):
        calls.append("load")
        return object()

    monkeypatch.setattr(
        model_module.mlflow.tracking, "MlflowClient", lambda: _FakeClient()
    )
    monkeypatch.setattr(model_module, "artifact_checksums", _fake_checksums)
    monkeypatch.setattr(model_module.mlflow.pyfunc, "load_model", _fake_load_model)


@pytest.fixture
def mlflow_calls(monkeypatch):
    calls: list[str] = []
    _stub_mlflow_surfaces(monkeypatch, calls)
    return calls


def _load(client, on_failure="raise"):
    from ario_mlflow.model import VerifiedModel

    return VerifiedModel(
        "models:/foo/1",
        asset_id="fraud-model",
        verify_status_client=client,
        on_failure=on_failure,
    )


# --- constructor validation --------------------------------------------------


def test_invalid_on_failure_rejected(mlflow_calls):
    from ario_mlflow.model import VerifiedModel

    with pytest.raises(ValueError):
        VerifiedModel("models:/foo/1", on_failure="explode")


def test_asset_id_and_client_must_come_together(mlflow_calls):
    from ario_mlflow.model import VerifiedModel

    with pytest.raises(ValueError):
        VerifiedModel("models:/foo/1", asset_id="fraud-model")
    with pytest.raises(ValueError):
        VerifiedModel(
            "models:/foo/1", verify_status_client=FakeVerifyStatusClient()
        )


def test_no_gate_kwargs_is_exactly_todays_behavior(mlflow_calls):
    from ario_mlflow.model import VerifiedModel

    vm = VerifiedModel("models:/foo/1")
    assert vm._artifact_verified is True
    assert mlflow_calls == ["integrity", "load"]


# --- gate ordering: verify-status first, fail fast ---------------------------


def test_gate_runs_before_integrity_and_load(mlflow_calls):
    client = FakeVerifyStatusClient(status=make_status(outcome="tampered"))
    with pytest.raises(AssetTamperedError):
        _load(client)
    assert client.calls == ["fraud-model"]
    # Fail fast: no artifact download/hash, no pyfunc load, no MLflow access.
    assert mlflow_calls == []


def test_verified_fresh_proceeds_through_both_gates(mlflow_calls):
    client = FakeVerifyStatusClient(status=make_status())
    vm = _load(client)
    assert vm._artifact_verified is True
    assert client.calls == ["fraud-model"]
    assert mlflow_calls == ["integrity", "load"]


# --- §9.1 outcome mapping -----------------------------------------------------


@pytest.mark.parametrize(
    ("outcome", "stale", "expected_exc"),
    [
        ("verified", True, AssetStaleError),
        ("tampered", False, AssetTamperedError),
        ("missing", False, AssetMissingError),
        ("unavailable", False, AssetStaleError),
        ("unknown", True, AssetUnknownError),
    ],
)
def test_outcome_mapping_per_contract(mlflow_calls, outcome, stale, expected_exc):
    client = FakeVerifyStatusClient(status=make_status(outcome=outcome, stale=stale))
    with pytest.raises(expected_exc) as exc_info:
        _load(client)
    assert exc_info.value.asset_id == "fraud-model"
    assert exc_info.value.status.outcome == outcome
    assert mlflow_calls == []


def test_gate_exceptions_are_one_family(mlflow_calls):
    client = FakeVerifyStatusClient(status=make_status(outcome="tampered"))
    with pytest.raises(AssetVerificationError):
        _load(client)


def test_integrity_error_joins_the_family():
    from ario_mlflow.model import IntegrityError

    assert issubclass(IntegrityError, AssetVerificationError)


# --- on_failure policies ------------------------------------------------------


def test_fail_closed_raises_like_raise(mlflow_calls):
    client = FakeVerifyStatusClient(status=make_status(outcome="missing"))
    with pytest.raises(AssetMissingError):
        _load(client, on_failure="fail_closed")
    assert mlflow_calls == []


def test_fail_open_logs_warning_and_proceeds(mlflow_calls, caplog):
    client = FakeVerifyStatusClient(status=make_status(outcome="tampered"))
    with caplog.at_level(logging.WARNING, logger="ario_mlflow.model"):
        vm = _load(client, on_failure="fail_open")
    assert mlflow_calls == ["integrity", "load"]
    [record] = [r for r in caplog.records if "fail_open" in r.message]
    assert record.levelno == logging.WARNING
    # Structured fields for SIEM pipelines. ``phase`` distinguishes the
    # load-time gate from the per-predict re-check (when enabled).
    assert record.ario_verify_status == {
        "asset_id": "fraud-model",
        "phase": "load",
        "error": "AssetTamperedError",
        "outcome": "tampered",
        "stale": False,
        "policy_hash": "a3f1b2c4",
        "current_tx_id": "TxAbc123",
    }


def test_transport_error_raises_under_default_policy(mlflow_calls):
    client = FakeVerifyStatusClient(
        exc=VerifyStatusTransportError("connection refused", asset_id="fraud-model")
    )
    with pytest.raises(VerifyStatusTransportError):
        _load(client)
    assert mlflow_calls == []


def test_transport_error_fail_open_proceeds_with_warning(mlflow_calls, caplog):
    client = FakeVerifyStatusClient(
        exc=VerifyStatusTransportError("connection refused", asset_id="fraud-model")
    )
    with caplog.at_level(logging.WARNING, logger="ario_mlflow.model"):
        vm = _load(client, on_failure="fail_open")
    assert mlflow_calls == ["integrity", "load"]
    [record] = [r for r in caplog.records if "fail_open" in r.message]
    assert record.ario_verify_status["error"] == "VerifyStatusTransportError"
    assert record.ario_verify_status["outcome"] is None  # no status received


# --- per-predict re-check (opt-in via recheck_per_predict=True) --------------


def _stub_for_predict(monkeypatch):
    """Stub the MLflow surfaces VerifiedModel touches during ``predict``.

    Returns a fake-anchor sentinel the caller can wire into the constructor.
    Mirrors the pattern from ``test_plugin_smoke.py``'s predict tests but
    skips registry resolution (mv lookup returns None so run_id stays
    ``"unknown"`` and no integrity check fires).
    """
    import ario_mlflow.model as model_module

    monkeypatch.setattr(model_module, "_resolve_model_version", lambda c, u: None)
    monkeypatch.setattr(
        model_module.mlflow.pyfunc,
        "load_model",
        lambda uri: type("M", (), {"predict": lambda self, x: [1]})(),
    )
    monkeypatch.setattr(
        model_module.mlflow.tracking, "MlflowClient", lambda: type("C", (), {})()
    )
    monkeypatch.setattr(
        model_module.mlflow, "get_active_trace_id", lambda: None, raising=False
    )

    class _FakeAnchor:
        enabled = False

        def upload_proof(self, env, *a, **kw):
            return None

    return _FakeAnchor()


def test_recheck_per_predict_requires_verify_status_client():
    """``recheck_per_predict=True`` alone is meaningless — should fail at
    construction with a clear ValueError, the same shape as the other
    "kwargs must come together" errors."""
    from ario_mlflow.model import VerifiedModel

    with pytest.raises(ValueError, match="recheck_per_predict"):
        VerifiedModel("models:/foo/1", recheck_per_predict=True)


def test_default_recheck_is_off_predict_does_not_call_client(monkeypatch):
    """Backward-compat: without ``recheck_per_predict=True``, the predict
    path makes ZERO verify-status calls — the legacy load-time-only gate
    behavior holds exactly as before."""
    from ario_mlflow.model import VerifiedModel

    client = FakeVerifyStatusClient(status=make_status())
    anchor = _stub_for_predict(monkeypatch)

    vm = VerifiedModel(
        "models:/foo/1",
        asset_id="fraud-model",
        verify_status_client=client,
        anchor=anchor,
    )
    # One call at __init__; nothing more.
    assert client.calls == ["fraud-model"]

    vm.predict([1.0, 2.0])
    assert client.calls == ["fraud-model"], (
        "predict() must not consult the agent unless recheck_per_predict=True"
    )


def test_recheck_per_predict_consults_client_each_call(monkeypatch):
    """``recheck_per_predict=True`` and ``recheck_max_cache_age=None``
    (the default) consult the agent on every ``predict()``. Fresh status
    is the §9.1 stance — caching ``outcome``/``stale`` for gating decisions
    is opt-in only."""
    from ario_mlflow.model import VerifiedModel

    client = FakeVerifyStatusClient(status=make_status())
    anchor = _stub_for_predict(monkeypatch)

    vm = VerifiedModel(
        "models:/foo/1",
        asset_id="fraud-model",
        verify_status_client=client,
        anchor=anchor,
        recheck_per_predict=True,
    )
    # One call from __init__, plus one per predict.
    vm.predict([1.0, 2.0])
    vm.predict([3.0, 4.0])
    assert client.calls == ["fraud-model", "fraud-model", "fraud-model"]


def test_recheck_per_predict_tampered_raises_before_inference(monkeypatch):
    """Tamper detected by the agent AFTER load — the per-predict gate
    refuses subsequent inference (under default ``on_failure="raise"``)."""
    from ario_mlflow.model import VerifiedModel

    client = FakeVerifyStatusClient(status=make_status())
    anchor = _stub_for_predict(monkeypatch)

    vm = VerifiedModel(
        "models:/foo/1",
        asset_id="fraud-model",
        verify_status_client=client,
        anchor=anchor,
        recheck_per_predict=True,
    )

    # Flip the agent's verdict to "tampered" between load and predict.
    client._status = make_status(outcome="tampered")

    with pytest.raises(AssetTamperedError):
        vm.predict([1.0, 2.0])


def test_recheck_per_predict_fail_open_logs_phase_predict(monkeypatch, caplog):
    """fail_open under per-predict gate proceeds with inference and logs
    a structured WARN tagged ``phase="predict"`` — distinct from the
    load-time gate's ``phase="load"`` so SIEM pipelines can route them
    separately."""
    from ario_mlflow.model import VerifiedModel

    client = FakeVerifyStatusClient(status=make_status())
    anchor = _stub_for_predict(monkeypatch)

    vm = VerifiedModel(
        "models:/foo/1",
        asset_id="fraud-model",
        verify_status_client=client,
        anchor=anchor,
        recheck_per_predict=True,
        on_failure="fail_open",
    )
    client._status = make_status(outcome="tampered")

    with caplog.at_level(logging.WARNING, logger="ario_mlflow.model"):
        result = vm.predict([1.0, 2.0])

    # Prediction proceeded.
    assert result.prediction == [1]
    # Structured log carries phase="predict" and the tampered outcome.
    [record] = [
        r for r in caplog.records
        if "fail_open" in r.message and r.ario_verify_status.get("phase") == "predict"
    ]
    assert record.ario_verify_status["error"] == "AssetTamperedError"
    assert record.ario_verify_status["outcome"] == "tampered"
    assert "prediction" in record.message  # phrasing reflects the phase


def test_recheck_per_predict_honors_max_cache_age(monkeypatch):
    """``recheck_max_cache_age`` flows straight into
    ``VerifyStatusClient.get(max_cache_age=...)`` — the §9.2 hot-path knob.
    The client's own cache is responsible for de-duping (tested in
    ``test_verify_status_client``); here we just pin the wiring."""
    from ario_mlflow.model import VerifiedModel

    received_kwargs: list[dict] = []

    class _RecordingClient:
        def __init__(self):
            self._status = make_status()
            self.calls: list[str] = []

        def get(self, asset_id, **kwargs):
            self.calls.append(asset_id)
            received_kwargs.append(kwargs)
            return self._status

    client = _RecordingClient()
    anchor = _stub_for_predict(monkeypatch)

    vm = VerifiedModel(
        "models:/foo/1",
        asset_id="fraud-model",
        verify_status_client=client,
        anchor=anchor,
        recheck_per_predict=True,
        recheck_max_cache_age=15.0,
    )
    vm.predict([1.0, 2.0])

    # Two gets: one at load (max_cache_age=None — fresh), one at predict
    # (max_cache_age=15.0 — opt-in §9.2 hot-path cache).
    assert [kw.get("max_cache_age") for kw in received_kwargs] == [None, 15.0]


def test_fail_open_does_not_swallow_integrity_error(monkeypatch, caplog):
    """on_failure applies to the verify-status gate only — a tampered
    artifact (IntegrityError) must still raise under fail_open."""
    import ario_mlflow.model as model_module
    from ario_mlflow.model import IntegrityError, VerifiedModel

    class _FakeRun:
        data = type("D", (), {"tags": {"ario.artifact_hash": "EXPECTED"}})()

    class _FakeMV:
        name = "foo"
        version = 1
        run_id = "run-xyz"
        source = "runs:/run-xyz/model"
        tags = {}

    class _FakeClient:
        def get_model_version(self, n, v):
            return _FakeMV()

        def get_run(self, rid):
            return _FakeRun()

    monkeypatch.setattr(
        model_module.mlflow.tracking, "MlflowClient", lambda: _FakeClient()
    )
    monkeypatch.setattr(
        model_module, "artifact_checksums", lambda *a, **kw: {"model/foo": "bad"}
    )
    monkeypatch.setattr(
        model_module.mlflow.pyfunc, "load_model", lambda uri: object()
    )

    client = FakeVerifyStatusClient(status=make_status())
    with pytest.raises(IntegrityError):
        VerifiedModel(
            "models:/foo/1",
            asset_id="fraud-model",
            verify_status_client=client,
            on_failure="fail_open",
        )
