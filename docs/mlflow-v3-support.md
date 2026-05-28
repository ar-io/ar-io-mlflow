# MLflow 2.x + 3.x support — status & plan

> **Status: the one real v3 fix shipped in 0.2.1; the rest is verification, not fixing.** Empirical testing against real MLflow 2.22 and 3.12 shows the component behaviors the plugin relies on **work on v3** — MLflow 3's OSS file store keeps far more `runs:/` back-compat than the "models are first-class, beyond run-centric" framing implies. The remaining work is **real-MLflow integration coverage of the full plugin flows** (registration/promotion, `VerifiedModel`, verify, dataset) and fixing only what those tests actually surface — expected to be small. The acceptance gate is the integration test on both majors, **not** the mocked unit suite.

## Sources (verified, not inferred)

- The installed **mlflow 2.22.5 and 3.12.0** packages (ground truth for behavior).
- **MLflow v3.1.0 release notes** (the `LoggedModel`-first-class change).
- `tests/test_mlflow3_integration.py` + a dedicated verification round (2026-05-28).

> **Lesson, recorded:** the docs/marketing framing overstated v3 breakage. An earlier analysis (and even an earlier draft of *this* plan) claimed breaks in `get_active_trace_id`, stages, model loading, and artifact download that **do not occur** on OSS file-store v3. Every row below is from running real MLflow. The discipline for the rest of this work: **test-first on real 2.x + 3.x; fix only confirmed-red; claim only what the matrix proves.**

## Verified behavior matrix (real MLflow 2.22 + 3.12)

| Behavior the plugin depends on | v2.22 | v3.12 | Notes |
|---|---|---|---|
| `mlflow.log-model.history` run tag | present | **gone** | The one real v3 change → the 0.2.1 fix below. |
| `_logged_model_paths()` artifact auto-resolution | ✓ | ✓ (fixed 0.2.1) | Reads `run.outputs.model_outputs` on v3. |
| `download_artifacts(run_id, <name>)` | ✓ | ✓ | Works for default + custom model names. |
| Run-level `log_artifacts` + download (the plugin's `ario/`) | ✓ | ✓ | Verify side's `ario/payload.json` is fine. |
| `create_model_version(source="runs:/<rid>/<name>")` | ✓ | ✓ | `mv.source` stays `runs:/…` (not rewritten). |
| `pyfunc.load_model("runs:/<rid>/<name>")` — **VerifiedModel's actual load path** | ✓ | ✓ | The real load path works on v3. |
| `pyfunc.load_model("models:/<name>/<version>")` for a `runs:/`-sourced version | ✓ | ✗ | Fails on v3 ("no MLmodel") — **but the plugin doesn't use this path** (it loads via `mv.source`). |
| `transition_model_version_stage` | ✓ | ✓ (deprecated) | Promotion path functions on v3. |
| `get_model_version` / `set`+`get_registered_model_alias` | ✓ | ✓ | |
| `mlflow.get_active_trace_id()` | ✓ | ✓ | Present + not deprecated in 3.12. |
| `run.inputs.dataset_inputs` | ✓ | ✓ | Dataset-anchor path's source is populated. |
| Filesystem tracking store (`./mlruns`, the default) | ✓ | ✓ (deprecated) | Deprecated upstream as of Feb 2026 → prefer `sqlite:///…`. |

**Takeaway:** the only v3-specific behavior change that touched the plugin was the dropped `log-model.history` tag (fixed in 0.2.1). Every other entry point uses `runs:/` resolution + run-level artifacts, which still work on v3. So the plugin **very likely already works on v3 across all entry points** — but only the artifact-resolution path has been *integration-verified*. Closing that verification gap is the plan.

## Plan

### Phase A — artifact auto-resolution + CI gate · ✅ done (0.2.1)
`_logged_model_paths()` reads `run.outputs.model_outputs` on v3; `tests/test_mlflow3_integration.py` + a CI `integration` job run on `mlflow<3` **and** `mlflow>=3`. Shipped to PyPI in 0.2.1.

### Phase B — integration coverage for the remaining flows (test-first; fix only confirmed failures)
Each item: write a real-MLflow integration test exercising the **plugin's full flow** (real tracking store + a stub/disabled `ArweaveAnchor` so no network), run on real 2.x + 3.x, fix only what's red.

| # | Flow | Test (extend `test_mlflow3_integration.py`) | Expected v3 risk | Watch for |
|---|------|---------------------------------------------|------------------|-----------|
| B1 | **Registration + promotion** (`ArioMlflowClient`) | `create_registered_model` → `ArioMlflowClient.create_model_version` (anchors) → `transition_model_version_stage` (anchors) → verify both proofs | **Low** (APIs verified) | `create_model_version` source handling on v3; multi-model-output disambiguation |
| B2 | **`VerifiedModel`** end-to-end | register → `VerifiedModel("models:/name/1")` → integrity check matches `ario.artifact_hash` → `predict()` → per-prediction anchor | **Low** (load + re-hash verified) | the `model_uri → mv.source → artifact_path` resolution under v3 |
| B3 | **Verify** end-to-end | `anchor()` → `verify_proof_by_tx` + `verify_record` for training / registration / prediction | **Low** (run-artifact download + `get_trace_info` verified) | source-of-truth refetch for registration/prediction on v3 |
| B4 | **Dataset anchoring** | `anchor(dataset=…)` + implicit in-training dataset anchoring | **Low** (`run.inputs.dataset_inputs` verified) | `_serialize_dataset_inputs` shape on v3 |

### Phase C — CI, docs, release · S
- Extend the `integration` CI job so B1–B4 run on `mlflow<3` **and** `mlflow>=3` (matrix already exists).
- Flip docs (README/CLAUDE.md) from "core path supported" → **"MLflow 2.x + 3.x fully supported."**
- **Release:** if B1–B4 surface **no behavior change** → patch (`0.2.2`, adds coverage + any tiny fixes). If a behavior change is required (e.g. translating a `runs:/` source to `models:/<model_id>` at registration) → minor (`0.3.0`). Cut via the proven Release → Trusted-Publishing flow.

### Phase D — future-proofing (deprecations, NOT current breaks) · defer
- **Aliases for promotion** — before MLflow removes stages.
- **`get_last_active_trace_id` shim** — when `get_active_trace_id` is removed.
- **sqlite backend guidance** in README/`plugin-production.md` (file store deprecated Feb 2026).

## Sequencing & effort

B1 → B2 → B3 → B4 (each independent; order is convenience), then Phase C. **~2–4 days, mostly test-writing** — fixes expected minimal given the verification matrix. Phase D deferred until upstream forces it.

## Acceptance criteria (v3 "fully supported")

1. Integration tests for **registration/promotion, `VerifiedModel`, verify, and dataset anchoring** pass on real MLflow 2.x **and** 3.x.
2. The CI `integration` matrix runs all of them on both majors.
3. Any behavior change required for v3 is version-conditional (keyed off the installed MLflow major) — v2 behavior unchanged.
4. Docs flip to full v2 + v3 parity; a release ships the new coverage.
