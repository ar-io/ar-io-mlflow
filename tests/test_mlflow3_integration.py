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


def _train_and_log_model(tmp_path) -> str:
    """Real run: train a tiny sklearn model and log it. Returns the run_id.

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
            mlflow.sklearn.log_model(clf, name="model")        # MLflow 3.x
        except TypeError:
            mlflow.sklearn.log_model(clf, artifact_path="model")  # MLflow 2.x
        return run.info.run_id


def test_report_mlflow_version():
    import mlflow

    print(f"\n[integration] MLflow under test: {mlflow.__version__} (major {_mlflow_major()})")


def test_artifact_checksums_resolves_logged_model(tmp_path):
    """Δ1 gate — the plugin must locate + hash a real logged model.

    The whole artifact-integrity guarantee depends on this. If it fails, the
    plugin cannot prove model integrity on this MLflow version.
    """
    from ario_mlflow.anchoring import artifact_checksums

    run_id = _train_and_log_model(tmp_path)
    print(f"\n[integration] major={_mlflow_major()} run_id={run_id}")

    checksums = artifact_checksums(run_id, artifact_path="model")
    print(f"[integration] model files hashed: {sorted(checksums)}")

    assert checksums, (
        "no model artifacts located/hashed — on MLflow 3.x this is the Δ1 break "
        "(models are no longer run artifacts; runs:/<run_id>/model can't find them)"
    )
    assert any(f.endswith("MLmodel") for f in checksums), "MLmodel descriptor not hashed"


def test_logged_model_paths_discovers_model(tmp_path):
    """The plugin auto-resolves the artifact path from the run's logged-model
    history tag. On v3 that tag is populated differently / not at all."""
    import mlflow
    from ario_mlflow.anchoring import _logged_model_paths

    run_id = _train_and_log_model(tmp_path)
    run = mlflow.tracking.MlflowClient().get_run(run_id)
    paths = _logged_model_paths(run)  # helper expects the Run object (reads .data.tags)
    print(f"\n[integration] major={_mlflow_major()} logged_model_paths={paths}")
    assert "model" in paths, (
        f"logged-model history did not surface the 'model' path (got {paths}) — "
        "auto-resolution of artifact_path is v2-shaped"
    )
