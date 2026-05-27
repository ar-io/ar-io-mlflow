"""Real-MLflow (NON-mocked) integration test — the MLflow-3 support gate.

Unlike the rest of the suite (which mocks every MLflow interaction), this
exercises the plugin against a **real** MLflow tracking store (file-based, in
``tmp_path``) with a **real** logged model. It is the acceptance gate for
MLflow 3.x support — see ``docs/mlflow-v3-support.md``.

Expected behavior today:
  - **MLflow 2.x:** passes (model is a run artifact; the plugin resolves +
    hashes it).
  - **MLflow 3.x:** EXPECTED TO FAIL on the artifact-resolution assertions —
    MLflow 3 moved model artifacts out of the run (first-class ``LoggedModel``),
    so ``runs:/<run_id>/model`` resolution can't locate them (Δ1). When the Δ1
    rework lands, this must go green on 3.x too.

Gated behind ``ARIO_MLFLOW_INTEGRATION=1`` so it never runs in the normal
(mocked) CI suite. Run:

    ARIO_MLFLOW_INTEGRATION=1 pytest tests/test_mlflow3_integration.py -v -s
"""

from __future__ import annotations

import os

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
