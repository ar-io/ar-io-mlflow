"""Real-MLflow (NON-mocked) integration test — the MLflow-3 support gate.

Unlike the rest of the suite (which mocks every MLflow interaction), this
exercises the plugin against a **real** MLflow tracking store (file-based, in
``tmp_path``) with a **real** logged model. It is the acceptance gate for
MLflow 3.x support — see ``docs/mlflow-v3-support.md``.

These tests must go **green on both MLflow 2.x and 3.x**. The only v3-specific
behavior change that touched the plugin (the dropped ``mlflow.log-model.history``
run tag → artifact auto-resolution) was fixed in 0.2.1; the verified behavior
matrix in ``docs/mlflow-v3-support.md`` shows every other entry point uses
``runs:/`` resolution + run-level artifacts, which still work on v3. The Phase B
tests below exist to *prove* that across the full plugin flows, not assume it.

Coverage:
  - ``_logged_model_paths`` / ``artifact_checksums`` — the v3 fix (Phase A).
  - ``ArioMlflowClient`` registration + promotion, ``VerifiedModel`` load +
    predict + tamper rejection, ``full_verify`` across event types, dataset
    anchoring (Phase B).
  - All four ``models:/`` URI forms through ``VerifiedModel``: numeric, alias,
    legacy stage (Python-side fallback on v3 where ``current_stage`` was
    dropped from the search grammar), and v3-native ``models:/<model_id>``
    (the LoggedModel direct form; v3-gated).
  - Multi-model-output guard in ``anchor()``, multi-dataset-input
    serialization, and the training-→-training chain via the
    ``log_model(registered_model_name=…)`` auto-register idiom.

No network: the ``ArioMlflowClient`` tests inject a stub anchor that returns
deterministic fake upload results, so the full tag-writing + chaining + artifact
path runs without touching Arweave/Turbo.

Gated behind ``ARIO_MLFLOW_INTEGRATION=1`` so it never runs in the normal
(mocked) CI suite. Run:

    ARIO_MLFLOW_INTEGRATION=1 pytest tests/test_mlflow3_integration.py -v -s
"""

from __future__ import annotations

import json
import os
import threading
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ARIO_MLFLOW_INTEGRATION") != "1",
    reason="real-MLflow integration test; set ARIO_MLFLOW_INTEGRATION=1 to run",
)


def _mlflow_major() -> int:
    import mlflow

    return int(mlflow.__version__.split(".")[0])


def _train_and_log_model(tmp_path, name: str = "fraud_clf") -> str:
    """Real run: train a tiny sklearn model and log it under ``name``.

    Defaults to a **non-default** name on purpose — that's the case that
    exposes the v3 auto-resolution gap (a model named ``"model"`` passes by
    coincidence via the ``anchor()`` fallback even when resolution is broken).
    Handles the v2/v3 ``log_model`` signature difference (v3 renamed
    ``artifact_path`` → ``name``).
    """
    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import LogisticRegression

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    with mlflow.start_run() as run:
        mlflow.log_param("max_iter", 100)
        mlflow.log_metric("accuracy", 0.9)
        clf = LogisticRegression(max_iter=100).fit([[0, 0], [1, 1], [0, 1], [1, 0]], [0, 1, 0, 1])
        try:
            mlflow.sklearn.log_model(clf, name=name)        # MLflow 3.x
        except TypeError:
            mlflow.sklearn.log_model(clf, artifact_path=name)  # MLflow 2.x
        return run.info.run_id


def test_report_mlflow_version():
    import mlflow

    print(f"\n[integration] MLflow under test: {mlflow.__version__} (major {_mlflow_major()})")


def test_logged_model_paths_resolves_custom_name(tmp_path):
    """The plugin must auto-resolve the artifact path of a logged model —
    on v2 from the ``mlflow.log-model.history`` tag, on v3 from
    ``run.outputs.model_outputs``. A non-default name is used deliberately:
    if resolution returns ``[]``, ``anchor()`` falls back to ``"model"`` and
    silently skips the hash for this model. This is the v3 gap the fix closes.
    """
    import mlflow
    from ario_mlflow.anchoring import _logged_model_paths

    run_id = _train_and_log_model(tmp_path, name="fraud_clf")
    run = mlflow.tracking.MlflowClient().get_run(run_id)
    paths = _logged_model_paths(run)  # helper takes the Run object
    print(f"\n[integration] major={_mlflow_major()} logged_model_paths={paths}")
    assert "fraud_clf" in paths, (
        f"auto-resolution did not surface the logged model name (got {paths}); "
        "on v3 this means _logged_model_paths isn't reading run.outputs.model_outputs"
    )


def test_custom_named_model_is_hashable_end_to_end(tmp_path):
    """The artifact-integrity guarantee for a custom-named model: the resolved
    path must let ``artifact_checksums`` hash the real model files."""
    import mlflow
    from ario_mlflow.anchoring import _logged_model_paths, artifact_checksums

    run_id = _train_and_log_model(tmp_path, name="fraud_clf")
    run = mlflow.tracking.MlflowClient().get_run(run_id)
    paths = _logged_model_paths(run)
    assert paths, "no logged-model path resolved (cannot hash the model)"

    checksums = artifact_checksums(run_id, artifact_path=paths[0])
    print(f"\n[integration] major={_mlflow_major()} hashed via '{paths[0]}': {sorted(checksums)}")
    assert any(f.endswith("MLmodel") for f in checksums), "MLmodel descriptor not hashed"


# --------------------------------------------------------------------------- #
# Phase B1 — ArioMlflowClient registration + promotion (full flow, real MLflow)
# --------------------------------------------------------------------------- #


class _StubAnchor:
    """Network-free anchor that returns deterministic fake upload results.

    ``enabled = True`` so the client takes the upload-success branch — it
    writes the ``ario.*_tx`` tags and follows the chain links — but
    ``upload_proof`` never touches Turbo/Arweave. Each call returns a unique
    ``tx_id`` so chain linkage is observable. Thread-safe: anchoring runs in
    daemon threads.
    """

    def __init__(self):
        self.enabled = True
        self.wallet_mode = "test-stub"
        self.wallet_type = "solana"
        self._lock = threading.Lock()
        self.uploaded: list[dict] = []

    def upload_proof(self, envelope: dict) -> dict:
        with self._lock:
            tx = f"STUBTX-{uuid.uuid4().hex}"
            self.uploaded.append(envelope)
        return {"tx_id": tx, "url": f"https://example.invalid/{tx}", "receipt": None}


def _train_log_and_anchor(tmp_path, name: str = "fraud_clf"):
    """Real run: train + log a model under ``name`` and anchor a training proof
    (with the stub anchor) so the run carries ``ario.training_tx`` /
    ``ario.artifact_hash`` for the registration chain + integrity check.

    Returns ``(run_id, source_uri, proof_engine, stub_anchor)``. The same
    ``proof_engine`` + ``stub_anchor`` are reused by the caller for the
    ``ArioMlflowClient`` so the whole chain shares one signing key.
    """
    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import LogisticRegression

    from ario_mlflow.anchoring import anchor
    from ario_mlflow.proof import ProofEngine

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    proof_engine = ProofEngine()
    stub = _StubAnchor()

    with mlflow.start_run() as run:
        mlflow.log_param("max_iter", 100)
        mlflow.log_metric("accuracy", 0.9)
        clf = LogisticRegression(max_iter=100).fit([[0, 0], [1, 1], [0, 1], [1, 0]], [0, 1, 0, 1])
        try:
            mlflow.sklearn.log_model(clf, name=name)        # MLflow 3.x
        except TypeError:
            mlflow.sklearn.log_model(clf, artifact_path=name)  # MLflow 2.x
        # artifact_path=None → exercises the same auto-resolution the v3 fix
        # repaired, end to end through the training proof. Dataset-input
        # anchoring is exercised separately (B4), so opt out of the
        # log_input requirement here.
        anchor(proof_engine=proof_engine, arweave=stub, allow_empty_dataset_inputs=True)
        run_id = run.info.run_id

    return run_id, f"runs:/{run_id}/{name}", proof_engine, stub


def _read_envelope(client, run_id: str, rel_path: str) -> dict:
    """Download an ``ario/...`` proof artifact from the run and parse it."""
    import mlflow

    local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=rel_path)
    with open(local) as f:
        return json.load(f)


def test_registration_and_promotion_full_flow(tmp_path):
    """Phase B1: ``create_registered_model`` → ``create_model_version`` (anchors)
    → ``transition_model_version_stage`` (anchors), verified end to end on real
    MLflow. Asserts the v3-sensitive surface the plan flagged:

    - registration re-hashes the model artifacts and matches the training
      ``ario.artifact_hash`` (``artifact_verified == "true"``) — proves
      ``create_model_version`` source handling + ``artifact_checksums`` resolve
      the same files on this major;
    - the chain links hold: registration ``previous_hash`` == training tx,
      promotion ``previous_hash`` == registration tx;
    - both signed envelopes verify against their embedded public key.
    """
    from ario_mlflow.client import ArioMlflowClient
    from ario_mlflow.proof import ProofEngine
    from ario_mlflow.verify import verify_signature

    run_id, source, proof_engine, stub = _train_log_and_anchor(tmp_path, name="fraud_clf")
    name = "fraud_clf_model"

    client = ArioMlflowClient(proof_engine=proof_engine, anchor=stub)
    client.create_registered_model(name)

    # --- registration -------------------------------------------------------
    mv = client.create_model_version(name, source, run_id=run_id)
    assert client.wait_for_anchor("registration", name, mv.version, timeout=30), \
        "registration anchor did not complete"
    reg_status = client.anchor_status("registration", name, mv.version)
    print(f"\n[integration] major={_mlflow_major()} registration status={reg_status}")
    assert reg_status["status"] == "anchored", reg_status
    assert reg_status["tx_id"], "no registration tx recorded"

    mv = client.get_model_version(name, mv.version)
    assert mv.source == source, f"v3 rewrote mv.source: {mv.source!r}"
    assert mv.tags.get("ario.registration_tx") == reg_status["tx_id"]
    assert mv.tags.get("ario.artifact_verified") == "true", (
        "registration artifact re-hash did not match the training ario.artifact_hash "
        f"(tags={dict(mv.tags)}) — artifact_checksums resolved different files on "
        f"major {_mlflow_major()}"
    )

    reg_env = _read_envelope(client, run_id, "ario/registration_proof.json")
    training_tx = client.get_run(run_id).data.tags.get("ario.training_tx")
    assert training_tx, "training proof did not record ario.training_tx"
    assert reg_env["previous_hash"] == training_tx, "registration not chained to training tx"
    assert verify_signature(reg_env, ProofEngine())["ok"], "registration signature invalid"

    # --- promotion ----------------------------------------------------------
    client.transition_model_version_stage(name, mv.version, "Production")
    assert client.wait_for_anchor("promotion", name, mv.version, timeout=30), \
        "promotion anchor did not complete"
    promo_status = client.anchor_status("promotion", name, mv.version)
    print(f"[integration] major={_mlflow_major()} promotion status={promo_status}")
    assert promo_status["status"] == "anchored", promo_status

    mv = client.get_model_version(name, mv.version)
    assert mv.tags.get("ario.promotion_tx") == promo_status["tx_id"]
    assert mv.tags.get("ario.promotion_payload_hash"), "missing ario.promotion_payload_hash"

    # promotion witness is keyed by event_id under ario/promotions/<event_id>/
    promo_env = next(e for e in stub.uploaded if e.get("event_type") == "stage_transition")
    promo_proof = _read_envelope(
        client, run_id, f"ario/promotions/{promo_env['event_id']}/proof.json"
    )
    assert promo_proof["previous_hash"] == mv.tags.get("ario.registration_tx"), \
        "promotion not chained to registration tx"
    assert verify_signature(promo_proof, ProofEngine())["ok"], "promotion signature invalid"


def _register(client, name: str, source: str, run_id: str):
    """Create the registered model + an anchored version, returning the version."""
    client.create_registered_model(name)
    mv = client.create_model_version(name, source, run_id=run_id)
    assert client.wait_for_anchor("registration", name, mv.version, timeout=30)
    assert client.anchor_status("registration", name, mv.version)["status"] == "anchored"
    return mv.version


# --------------------------------------------------------------------------- #
# Phase B2 — VerifiedModel end to end (load-time integrity + predict anchoring)
# --------------------------------------------------------------------------- #


def test_verified_model_predict_full_flow(tmp_path):
    """Phase B2: register → ``VerifiedModel("models:/name/v")`` → integrity check
    matches the training ``ario.artifact_hash`` → ``predict()`` anchors a
    per-prediction proof chained to the registration.

    The v3-sensitive surface (per the plan): the ``model_uri → mv.source →
    artifact_path`` resolution. ``VerifiedModel`` loads via ``mv.source``
    (``runs:/…``, which works on v3), NOT ``models:/name/v`` (which doesn't),
    and re-hashes the same files the training proof committed to.
    """
    from ario_mlflow.client import ArioMlflowClient
    from ario_mlflow.model import VerifiedModel
    from ario_mlflow.proof import ProofEngine
    from ario_mlflow.verify import verify_signature

    run_id, source, proof_engine, stub = _train_log_and_anchor(tmp_path, name="fraud_clf")
    name = "fraud_clf_served"

    client = ArioMlflowClient(proof_engine=proof_engine, anchor=stub)
    version = _register(client, name, source, run_id)
    registration_tx = client.get_model_version(name, version).tags.get("ario.registration_tx")
    assert registration_tx

    vm = VerifiedModel(f"models:/{name}/{version}", proof_engine=proof_engine, anchor=stub)
    print(f"\n[integration] major={_mlflow_major()} artifact_verified={vm._artifact_verified}")
    assert vm._artifact_verified is True, (
        "load-time integrity check did not pass — the mv.source re-hash did not "
        f"match the training ario.artifact_hash on major {_mlflow_major()}"
    )
    assert vm.run_id == run_id
    assert vm._prediction_previous_hash == registration_tx, \
        "predictions did not pick up the registration tx as chain head"

    result = vm.predict([0, 1])
    assert result.wait_for_anchor(timeout=30), "prediction anchor did not complete"
    print(f"[integration] major={_mlflow_major()} prediction status={result.proof_status}")
    assert result.proof_status == "anchored", result.anchor_error
    assert result.tx_id

    pred_proof = _read_envelope(
        client, run_id, f"ario/predictions/{result.decision_id}/proof.json"
    )
    assert pred_proof["previous_hash"] == registration_tx, \
        "prediction not chained to registration tx"
    assert pred_proof["event_type"] == "prediction"
    assert verify_signature(pred_proof, ProofEngine())["ok"], "prediction signature invalid"


def test_verified_model_rejects_tampered_artifact(tmp_path):
    """Phase B2: the load-time integrity guarantee must hold on both majors —
    if the run's ``ario.artifact_hash`` no longer matches the artifacts,
    ``VerifiedModel`` raises ``IntegrityError`` *before* loading user code.

    Simulated by corrupting the anchored hash (equivalent to the artifact
    being swapped under a fixed anchored hash). Proves the v3 re-hash path
    actually feeds the comparison rather than silently passing.
    """
    from ario_mlflow.client import ArioMlflowClient
    from ario_mlflow.model import IntegrityError, VerifiedModel

    run_id, source, proof_engine, stub = _train_log_and_anchor(tmp_path, name="fraud_clf")
    name = "fraud_clf_tampered"

    client = ArioMlflowClient(proof_engine=proof_engine, anchor=stub)
    version = _register(client, name, source, run_id)

    # Corrupt the anchored hash on the source run, then attempt to load.
    client.set_tag(run_id, "ario.artifact_hash", "deadbeef" * 8)
    with pytest.raises(IntegrityError):
        VerifiedModel(f"models:/{name}/{version}", proof_engine=proof_engine, anchor=stub)
    print(f"\n[integration] major={_mlflow_major()} IntegrityError raised as expected")


# --------------------------------------------------------------------------- #
# Phase B3 — verify checks against live MLflow (signature + anchored bytes +
# source-of-truth re-derivation) for training / registration / prediction
# --------------------------------------------------------------------------- #


def test_verify_full_chain_against_live_mlflow(tmp_path):
    """Phase B3: ``full_verify`` (checks 1–3, no Arweave fetch) must pass on real
    MLflow for **training, registration, and prediction** envelopes. This is the
    v3-sensitive verify surface: check 2 downloads ``ario/payload.json`` from the
    store, and check 3 re-derives the canonical bytes from the *live* MLflow
    surface — run params/metrics (training), model-version state (registration),
    and the ``ario.payload_json`` trace tag via ``get_trace_info`` (prediction,
    the MLflow-3 path that the 0.1.0 fix put on ``_tracing_client``).

    Uses the envelopes captured by the stub anchor and the artifacts/tags the
    plugin actually wrote — i.e. the operator verify flow minus the network fetch.
    """
    import mlflow

    from ario_mlflow.client import ArioMlflowClient
    from ario_mlflow.model import VerifiedModel
    from ario_mlflow.proof import ProofEngine
    from ario_mlflow.verify import full_verify

    run_id, source, proof_engine, stub = _train_log_and_anchor(tmp_path, name="fraud_clf")
    name = "fraud_clf_verify"

    client = ArioMlflowClient(proof_engine=proof_engine, anchor=stub)
    version = _register(client, name, source, run_id)

    vm = VerifiedModel(f"models:/{name}/{version}", proof_engine=proof_engine, anchor=stub)
    result = vm.predict([0, 1])
    assert result.wait_for_anchor(timeout=30)
    assert result.proof_status == "anchored", result.anchor_error

    verifier = mlflow.tracking.MlflowClient()
    by_type = {e["event_type"]: e for e in stub.uploaded}
    for event_type in ("training_complete", "model_registered", "prediction"):
        env = by_type.get(event_type)
        assert env is not None, f"no {event_type} envelope captured"
        res = full_verify(env, proof_engine=ProofEngine(), mlflow_client=verifier)
        print(
            f"\n[integration] major={_mlflow_major()} {event_type}: "
            f"sig={res['signature']['ok']} bytes={res['anchored_bytes']['ok']} "
            f"sot={res['source_of_truth']['ok']} overall={res['overall']}"
        )
        assert res["signature"]["ok"] is True, f"{event_type} signature: {res['signature']}"
        assert res["anchored_bytes"]["ok"] is True, (
            f"{event_type} anchored-bytes check failed on major {_mlflow_major()}: "
            f"{res['anchored_bytes']}"
        )
        assert res["source_of_truth"]["ok"] is True, (
            f"{event_type} source-of-truth re-derivation failed on major "
            f"{_mlflow_major()}: {res['source_of_truth']}"
        )
        assert res["overall"] is True, f"{event_type} overall: {res}"


# --------------------------------------------------------------------------- #
# Phase B4 — dataset anchoring (in-training + standalone) on real MLflow
# --------------------------------------------------------------------------- #


def _train_log_anchor_with_dataset(tmp_path, name: str = "fraud_clf"):
    """Real run: train + log a model AND a real ``mlflow.data`` dataset input,
    then anchor. Returns ``(run_id, dataset_entity, proof_engine, stub)``.

    The dataset entity returned here is the one read from
    ``run.inputs.dataset_inputs[0].dataset`` — the shape ``anchor(dataset=…)``
    is designed for. (Live ``mlflow.data.from_numpy(...)`` objects expose a
    different attribute shape — a pre-existing inconsistency, not a v3
    regression, so not in scope here.)
    """
    import mlflow
    import mlflow.data
    import mlflow.sklearn
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from ario_mlflow.anchoring import anchor
    from ario_mlflow.proof import ProofEngine

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("b4")
    proof_engine = ProofEngine()
    stub = _StubAnchor()

    ds = mlflow.data.from_numpy(
        np.array([[0, 0], [1, 1], [0, 1], [1, 0]]),
        targets=np.array([0, 1, 0, 1]),
        source="fraud-train-q1.csv",
        name="fraud_train_q1",
    )
    with mlflow.start_run() as run:
        mlflow.log_input(ds, context="training")
        mlflow.log_param("max_iter", 100)
        mlflow.log_metric("accuracy", 0.9)
        clf = LogisticRegression(max_iter=100).fit([[0, 0], [1, 1], [0, 1], [1, 0]], [0, 1, 0, 1])
        try:
            mlflow.sklearn.log_model(clf, name=name)
        except TypeError:
            mlflow.sklearn.log_model(clf, artifact_path=name)
        anchor(proof_engine=proof_engine, arweave=stub)
        run_id = run.info.run_id

    run_data = mlflow.tracking.MlflowClient().get_run(run_id)
    entity = run_data.inputs.dataset_inputs[0].dataset
    return run_id, entity, proof_engine, stub


def test_in_training_dataset_anchoring(tmp_path):
    """Phase B4 — in-training: with ``mlflow.log_input(ds, ...)`` set on the
    run, ``anchor()`` must (a) auto-anchor a standalone dataset event for
    each input (event_type=``dataset`` envelopes uploaded), and (b) include
    a ``dataset_inputs`` array in the training payload that round-trips
    through ``verify_source_of_truth`` — the live refetcher re-runs
    ``_serialize_dataset_inputs`` against ``run.inputs.dataset_inputs`` on the
    current major and the result must match the anchored bytes byte-for-byte.
    """
    import mlflow

    from ario_mlflow.proof import ProofEngine
    from ario_mlflow.verify import full_verify

    run_id, _entity, _pe, stub = _train_log_anchor_with_dataset(tmp_path)

    dataset_envs = [e for e in stub.uploaded if e["event_type"] == "dataset"]
    training_env = next(e for e in stub.uploaded if e["event_type"] == "training_complete")
    print(
        f"\n[integration] major={_mlflow_major()} dataset_envelopes_anchored="
        f"{len(dataset_envs)}"
    )
    assert dataset_envs, "no dataset envelope auto-anchored during training"

    verifier = mlflow.tracking.MlflowClient()
    res = full_verify(training_env, proof_engine=ProofEngine(), mlflow_client=verifier)
    print(
        f"[integration] major={_mlflow_major()} training-w-dataset: "
        f"sig={res['signature']['ok']} bytes={res['anchored_bytes']['ok']} "
        f"sot={res['source_of_truth']['ok']} overall={res['overall']}"
    )
    assert res["overall"] is True, (
        f"training verify failed with a logged dataset input on major "
        f"{_mlflow_major()} — _serialize_dataset_inputs drift between anchor "
        f"and verify? {res}"
    )

    # Check the verify side actually went through the dataset_inputs path
    # (vs. a payload that omitted the field entirely, which would also
    # trivially pass).
    payload = json.loads(res["anchored_bytes"]["payload_bytes"])
    assert payload.get("dataset_inputs"), "training payload missing dataset_inputs"


def test_standalone_dataset_event_signed_and_verifiable(tmp_path):
    """Phase B4 — standalone: ``anchor(dataset=entity)`` mints a signed
    ``event_type=dataset`` envelope with no active run required. Asserts the
    envelope shape + signature on both majors, using the entity shape the
    function is designed for (read from ``run.inputs.dataset_inputs``).
    """
    from ario_mlflow.anchoring import anchor
    from ario_mlflow.proof import ProofEngine
    from ario_mlflow.verify import verify_signature

    _run_id, entity, proof_engine, stub = _train_log_anchor_with_dataset(tmp_path)
    result = anchor(dataset=entity, proof_engine=proof_engine, arweave=stub)
    env = result["envelope"]
    print(
        f"\n[integration] major={_mlflow_major()} standalone dataset event_type="
        f"{env['event_type']} digest={result['payload']['digest']}"
    )
    assert env["event_type"] == "dataset"
    assert result["payload"]["name"] == "fraud_train_q1"
    assert result["payload"]["source_type"] == "local"
    assert verify_signature(env, ProofEngine())["ok"]


# --------------------------------------------------------------------------- #
# Additional v3 surface coverage (URI forms, multi-model, multi-dataset,
# training-→-training chain). These exist because they were probed once and
# then forgotten; the integration suite is the place to nail them down.
# --------------------------------------------------------------------------- #


def _setup_registered(tmp_path):
    """Train + log + anchor + register + alias + transition to Production.

    Returns ``(client, name, version, run_id, info, stub)``. ``info`` is the
    ``ModelInfo`` returned by ``log_model`` so v3-only callers can read
    ``info.model_id``.
    """
    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import LogisticRegression

    from ario_mlflow.anchoring import anchor
    from ario_mlflow.client import ArioMlflowClient
    from ario_mlflow.proof import ProofEngine

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("uris")
    pe, stub = ProofEngine(), _StubAnchor()
    with mlflow.start_run() as run:
        clf = LogisticRegression(max_iter=20).fit([[0, 0], [1, 1]], [0, 1])
        try:
            info = mlflow.sklearn.log_model(clf, name="m")
        except TypeError:
            info = mlflow.sklearn.log_model(clf, artifact_path="m")
        anchor(proof_engine=pe, arweave=stub, allow_empty_dataset_inputs=True)
        run_id = run.info.run_id

    name = "URI_M"
    client = ArioMlflowClient(proof_engine=pe, anchor=stub)
    client.create_registered_model(name)
    mv = client.create_model_version(name, f"runs:/{run_id}/m", run_id=run_id)
    client.wait_for_anchor("registration", name, mv.version, timeout=30)
    client.set_registered_model_alias(name, "champion", mv.version)
    client.transition_model_version_stage(name, mv.version, "Production")
    client.wait_for_anchor("promotion", name, mv.version, timeout=30)
    return client, name, mv.version, run_id, info, stub, pe


def test_verified_model_alias_uri(tmp_path):
    """`models:/<name>@<alias>` — v3-native idiom, must integrity-verify on
    both majors. Aliases are how v3 users will promote models once stages are
    fully gone; this test stays green as MLflow deprecates further."""
    from ario_mlflow.model import VerifiedModel

    client, name, version, run_id, _info, stub, pe = _setup_registered(tmp_path)
    vm = VerifiedModel(f"models:/{name}@champion", proof_engine=pe, anchor=stub)
    print(f"\n[integration] major={_mlflow_major()} alias artifact_verified={vm._artifact_verified}")
    assert vm._artifact_verified is True
    assert vm.run_id == run_id


def test_verified_model_legacy_stage_uri(tmp_path):
    """`models:/<name>/<stage>` — legacy v2 idiom. Native search works on v2;
    on v3 the search grammar dropped ``current_stage`` so the resolver falls
    back to a Python-side filter. End-to-end integrity check must hold on
    both majors so v2 codebases can upgrade to MLflow 3 without rewriting
    every load URI in the same change."""
    from ario_mlflow.model import VerifiedModel

    client, name, version, run_id, _info, stub, pe = _setup_registered(tmp_path)
    vm = VerifiedModel(f"models:/{name}/Production", proof_engine=pe, anchor=stub)
    print(
        f"\n[integration] major={_mlflow_major()} stage URI artifact_verified="
        f"{vm._artifact_verified}"
    )
    assert vm._artifact_verified is True
    assert vm.run_id == run_id


def test_verified_model_logged_model_id_uri_v3(tmp_path):
    """`models:/<model_id>` — v3-native LoggedModel direct. Before the
    resolver learned about LoggedModel ids, ``VerifiedModel`` silently
    degraded on this URI (no integrity check, GENESIS chain). Skipped on v2
    where ``ModelInfo.model_id`` doesn't exist."""
    if _mlflow_major() < 3:
        pytest.skip("models:/<model_id> is a v3-only URI form")
    from ario_mlflow.model import VerifiedModel

    client, _name, _version, run_id, info, stub, pe = _setup_registered(tmp_path)
    lm_id = info.model_id
    vm = VerifiedModel(f"models:/{lm_id}", proof_engine=pe, anchor=stub)
    print(
        f"\n[integration] major={_mlflow_major()} models:/<model_id> "
        f"run_id={vm.run_id} artifact_verified={vm._artifact_verified}"
    )
    assert vm._artifact_verified is True, (
        "LoggedModel-direct integrity verification did not run — the resolver "
        "regressed and is treating models:/<model_id> as unresolvable again"
    )
    assert vm.run_id == run_id


def test_anchor_raises_on_multi_model_run_without_explicit_path(tmp_path):
    """`_logged_model_paths()` enumerates every logged model — when a run has
    >1, ``anchor()`` must raise ``ValueError`` rather than silently picking
    one. Verifies the disambiguation guard on both majors and that the v3
    `run.outputs.model_outputs` path still produces the full list."""
    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import LogisticRegression

    from ario_mlflow.anchoring import _logged_model_paths, anchor
    from ario_mlflow.proof import ProofEngine

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("multi-model")
    pe, stub = ProofEngine(), _StubAnchor()
    with mlflow.start_run() as run:
        clf = LogisticRegression(max_iter=20).fit([[0, 0], [1, 1]], [0, 1])
        for n in ("model_a", "model_b"):
            try:
                mlflow.sklearn.log_model(clf, name=n)
            except TypeError:
                mlflow.sklearn.log_model(clf, artifact_path=n)
        rid = run.info.run_id

        r = mlflow.tracking.MlflowClient().get_run(rid)
        paths = set(_logged_model_paths(r))
        print(f"\n[integration] major={_mlflow_major()} multi-model paths={paths}")
        assert paths == {"model_a", "model_b"}

        with pytest.raises(ValueError, match="multiple model artifact paths"):
            anchor(proof_engine=pe, arweave=stub, allow_empty_dataset_inputs=True)

        # Explicit artifact_path disambiguates and anchors successfully.
        res = anchor(
            proof_engine=pe, arweave=stub,
            artifact_path="model_b", allow_empty_dataset_inputs=True,
        )
        assert res["envelope"]["event_type"] == "training_complete"


def test_anchor_with_multi_dataset_inputs(tmp_path):
    """A training run with multiple ``log_input`` calls must serialize every
    dataset into the canonical payload (sorted deterministically by
    ``_serialize_dataset_inputs``) and full_verify must re-derive the same
    list from live MLflow on both majors — proves the v3
    ``run.inputs.dataset_inputs`` shape round-trips identically."""
    import mlflow
    import mlflow.data
    import mlflow.sklearn
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from ario_mlflow.anchoring import anchor
    from ario_mlflow.proof import ProofEngine
    from ario_mlflow.verify import full_verify

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("multi-ds")
    pe, stub = ProofEngine(), _StubAnchor()
    ds_train = mlflow.data.from_numpy(
        np.array([[0, 0], [1, 1]]), targets=np.array([0, 1]),
        source="train.csv", name="train_q1",
    )
    ds_val = mlflow.data.from_numpy(
        np.array([[0, 1], [1, 0]]), targets=np.array([0, 1]),
        source="val.csv", name="val_q1",
    )
    with mlflow.start_run() as run:
        mlflow.log_input(ds_train, context="training")
        mlflow.log_input(ds_val, context="validation")
        clf = LogisticRegression(max_iter=20).fit([[0, 0], [1, 1]], [0, 1])
        try:
            mlflow.sklearn.log_model(clf, name="m")
        except TypeError:
            mlflow.sklearn.log_model(clf, artifact_path="m")
        anchor(proof_engine=pe, arweave=stub)
        run_id = run.info.run_id

    training_env = next(e for e in stub.uploaded if e["event_type"] == "training_complete")
    dataset_envs = [e for e in stub.uploaded if e["event_type"] == "dataset"]
    print(
        f"\n[integration] major={_mlflow_major()} dataset_events={len(dataset_envs)}"
    )
    assert len(dataset_envs) == 2, (
        f"expected one dataset envelope per logged input, got {len(dataset_envs)}"
    )

    res = full_verify(training_env, proof_engine=ProofEngine(),
                      mlflow_client=mlflow.tracking.MlflowClient())
    assert res["overall"] is True, res
    payload = json.loads(res["anchored_bytes"]["payload_bytes"])
    names = sorted(d["name"] for d in payload["dataset_inputs"])
    assert names == ["train_q1", "val_q1"], (
        f"multi-dataset payload didn't include both inputs on major "
        f"{_mlflow_major()}: {names}"
    )


def test_training_to_training_chain_via_auto_register(tmp_path):
    """The training-→-training chain (``ario.last_training_hash`` on the
    registered model) only fires when a model version exists for the current
    run at ``anchor()`` time. The realistic path that produces this is
    ``mlflow.<flavor>.log_model(name=…, registered_model_name=…)`` (the
    auto-register-at-log-time idiom) — the README's separate-register flow
    cannot chain training-→-training because no version exists yet when
    ``anchor()`` runs. This test pins the auto-register path on both majors."""
    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import LogisticRegression

    from ario_mlflow.anchoring import TAG_LAST_TRAINING_HASH, anchor
    from ario_mlflow.proof import ProofEngine

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("chain")
    pe, stub = ProofEngine(), _StubAnchor()
    rm_name = "ChainModel"

    def _train_round(round_idx: int):
        with mlflow.start_run() as run:
            clf = LogisticRegression(max_iter=20).fit(
                [[0, round_idx % 2], [1, 1]], [0, 1],
            )
            mlflow.log_param("round", round_idx)
            try:
                mlflow.sklearn.log_model(clf, name="m", registered_model_name=rm_name)
            except TypeError:
                mlflow.sklearn.log_model(
                    clf, artifact_path="m", registered_model_name=rm_name,
                )
            res = anchor(proof_engine=pe, arweave=stub, allow_empty_dataset_inputs=True)
        return res

    r1 = _train_round(1)
    r2 = _train_round(2)

    mc = mlflow.tracking.MlflowClient()
    head_after_r2 = mc.get_registered_model(rm_name).tags.get(TAG_LAST_TRAINING_HASH)
    print(
        f"\n[integration] major={_mlflow_major()} chain: r1.payload_hash="
        f"{r1['payload_hash'][:16]}.. r2.previous_hash="
        f"{r2['envelope']['previous_hash'][:16]}.. head={head_after_r2[:16] if head_after_r2 else None}"
    )
    assert r2["envelope"]["previous_hash"] == r1["payload_hash"], (
        "second training run did not chain to the first via ario.last_training_hash"
    )
    assert head_after_r2 == r2["payload_hash"], (
        "ario.last_training_hash on registered model not updated after r2"
    )
