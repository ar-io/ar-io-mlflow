# MLflow 3.x support — plan (dual v2/v3)

> **Status: the v3 model-resolution fix has landed and is verified** on real MLflow **2.22 + 3.12** (`tests/test_mlflow3_integration.py`, now run in CI on both majors). The core anchor/verify artifact-hashing path works on both. **Remaining for full declared dual support:** real-MLflow-3 integration coverage for the registration/promotion (`ArioMlflowClient`), prediction (`VerifiedModel.predict`), and full `verify_record` paths (they pass the *mocked* suite on 3.x but lack dedicated v3 coverage); plus sqlite-backend docs (the file store is deprecated). The acceptance gate is the integration test, **not** the mocked unit suite.

## Sources (authoritative — verified, not inferred)

- **The installed `mlflow==3.12` package** (ground truth for behavior).
- **MLflow v3.1.0 release notes** ("MLflow 3", 2025-06-10): *"the new `LoggedModel` entity as a first-class citizen, moving beyond the traditional run-centric approach."* Its breaking-changes list does **not** include removal of `get_active_trace_id` or `transition_model_version_stage`.
- **`tests/test_mlflow3_integration.py`** run against real MLflow 2.x and 3.12.

> An earlier docs/web-search-based draft of this analysis overstated the breakage (claimed `get_active_trace_id` removed and `runs:/` model resolution fully broken). The empirical run + installed-package + official release notes corrected it; this doc reflects the corrected, sourced picture.

## What actually changed in MLflow 3 (and what it does to us)

### The one real functional gap — model auto-resolution

MLflow 3 makes models **first-class `LoggedModel` entities** (model-centric, not run-centric). Concretely, verified on 3.12:

- `mlflow.<flavor>.log_model(..., name="x")` returns a `ModelInfo` with `model_uri = models:/<model_id>`; artifacts are stored under `<store>/models/<model_id>/artifacts`, **not** under the run.
- The **`mlflow.log-model.history` run tag is gone.** The logged model is now recorded in **`run.outputs.model_outputs`** (`LoggedModelOutput(model_id=...)`) and queryable via `client.search_logged_models()`.

Impact on the plugin: `_logged_model_paths(run)` (anchoring.py) reads `mlflow.log-model.history` → returns `[]` on v3. Consequence:
- Model logged with the **default `name="model"`** → `anchor()` falls back to `"model"`, `artifact_checksums(run_id, "model")` still resolves on 3.12 → **hashes fine.** ✓ (empirically passes)
- Model logged under a **non-default name** → auto-resolution returns `[]`, `anchor()` falls back to `"model"`, which doesn't match → `ArtifactAccessError` → artifact hash **silently skipped** (`artifact_status="hash_failed"`). ✗

So the integrity guarantee silently degrades for custom-named models on v3. `VerifiedModel`'s load-time re-hash uses the `models:/` URI path and needs the same v3-aware resolution check.

**Fix:** make `_logged_model_paths` (and the verify-side refetcher) read `run.outputs.model_outputs` → resolve the `LoggedModel` / `models:/<model_id>` on v3, falling back to the `mlflow.log-model.history` tag on v2. This is the bulk of "support v3."

### Not breaks (corrected — do NOT spend effort here)

- **`mlflow.get_active_trace_id()`** — present and not deprecated in 3.12 (verified in the installed source; absent from the v3 breaking-changes list). The plugin's two call sites work. *No change needed now.*
- **`transition_model_version_stage` / model stages** — present in 3.12, deprecated-but-functional. The promotion integration point still works (with warnings). Migrating to aliases is good hygiene but **not required** for v3 support.

### Deprecation-driven future risk (monitor, not urgent)

- **Filesystem tracking backend (`./mlruns`) deprecated as of Feb 2026** (installed-package warning). This is the plugin/CLI **default** (`MLFLOW_TRACKING_URI` default `./mlruns`). Recommend documenting a sqlite/db backend for new users; no code change forced yet.
- Stages and possibly `get_active_trace_id` may be removed in a *later* MLflow major — revisit then; the spec-versioned design absorbs it.

## What does NOT break (scope guardrails)

- The plugin's own `ario/*.json` + `verification.html` are logged as **run** artifacts (`log_artifacts`) — still works in v3.
- Crypto core (`proof.py`), Arweave/Turbo upload, `verify_record`, envelope spec — MLflow-agnostic.
- `download_artifacts(run_id, "model")`, `get_run`, `search_model_versions`, `get_model_version_by_alias`, `set_tag`, `set_trace_tag`, `get_trace_info` — all work on 3.12.

## Plan (sequenced)

| # | Step | Effort | Notes |
|---|------|--------|-------|
| 0 | **Honest docs** | ✅ done | CLAUDE.md/README corrected to the sourced reality. |
| 1 | **Real-MLflow integration test** | ✅ done | `tests/test_mlflow3_integration.py` — passes on real 2.22 + 3.12; uses a non-default model name to exercise the v3 gap. The acceptance gate. |
| 2 | **v3-aware model resolution** | ✅ done | `_logged_model_paths` reads `run.outputs.model_outputs` → `get_logged_model(model_id).name` on v3, falling back to `mlflow.log-model.history` on v2. Feeds both `anchor()` and the verify-side refetcher (one helper, both paths). |
| 3 | **CI matrix** | ✅ done | `test.yml` `integration` job runs the gate on `mlflow<3` **and** `mlflow>=3`. |
| 4 | **Broaden v3 integration coverage** | M | Add real-MLflow-3 tests for registration/promotion (`ArioMlflowClient`), prediction (`VerifiedModel.predict`), and full `verify_record` — currently mocked-only on 3.x. Required before claiming *full* v3 parity. |
| 5 | **Docs: sqlite backend guidance** | S | Note the file-store deprecation (Feb 2026); recommend `sqlite:///…` for new setups. |
| 6 | *(later, optional)* aliases for promotion; `get_last_active_trace_id` shim | S each | Only when MLflow actually removes stages / `get_active_trace_id`. |

**Landed: the artifact-resolution path works + is CI-tested on v2 + v3.** Remaining (#4) is broadening integration coverage to the other entry points before declaring full parity.

## Decisions

- **Dual support via version detection** (`mlflow.__version__` major), contained in the model-resolution helper(s). Don't drop v2.
- **Do not cap the pin** (`mlflow>=2.14.0` stays). Capping `<3` would force-downgrade users where v3 already mostly works.
- **The mocked unit suite is not the v3 gate** — the integration test is.

## Acceptance criteria (v3 "done")

1. `tests/test_mlflow3_integration.py` passes on real MLflow 2.x **and** 3.x, including a model logged under a **non-default name** (`anchor()` produces a non-empty `artifact_hash`; `VerifiedModel` re-hash matches; `verify_record` passes).
2. CI runs the integration test on a 2.x **and** a 3.x job.
3. Docs flip from "v2 supported, v3 largely works" to "v2 + v3 supported," with the file-store guidance noted.
