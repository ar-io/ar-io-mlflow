# MLflow 2.x + 3.x support — status & verified history

> **Status: MLflow 2.x and 3.x are both fully supported.** Every plugin flow is integration-tested on real MLflow at the **boundary versions of each major** — `==2.14.*` (the pyproject floor), `==2.22.*` (latest 2.x), `==3.0.*` (the first release where `LoggedModel` became first-class), and `==3.12.*` (latest 3.x) — and CI runs both the mocked unit suite (now version-clean across all four) and the real-MLflow integration gate on every matrix entry. The v3 artifact-resolution fix shipped in 0.2.1; Phase B integration coverage and the two MLflow-2.x prediction-verification bug fixes shipped in 0.2.2; the post-audit URI-resolution fixes (legacy stage on v3, v3-native LoggedModel direct) shipped in 0.2.3; the cross-version mocked-suite cleanup + boundary CI matrix + live-network smoke shipped in 0.2.4.

## Sources (verified, not inferred)

- The installed **mlflow 2.22.5 and 3.12.0** packages (ground truth for behavior).
- **MLflow v3.1.0 release notes** (the `LoggedModel`-first-class change).
- `tests/test_mlflow3_integration.py` — the real-MLflow integration gate, run on both majors in CI.

> **Lesson, recorded:** the docs/marketing framing initially overstated v3 breakage. An earlier draft of this plan claimed breaks in `get_active_trace_id`, stages, model loading, and artifact download that do not occur on OSS file-store v3. Empirical testing against real MLflow corrected that. Then Phase B coverage on real 2.22 surfaced the *inverse* surprise: **two top-level APIs the predict path used (`mlflow.get_active_trace_id`, `mlflow.set_trace_tag`) don't exist on 2.x at all** — prediction source-of-truth verification was silently broken on 2.x despite docs claiming otherwise. Both rows below are now from running real MLflow on both majors and asserting full end-to-end verify.

## Verified behavior matrix (real MLflow 2.22 + 3.12)

| Behavior the plugin depends on | v2.22 | v3.12 | Notes |
|---|---|---|---|
| `mlflow.log-model.history` run tag | present | **gone** | The v3 change → fixed in 0.2.1 (read `run.outputs.model_outputs` on v3). |
| `_logged_model_paths()` artifact auto-resolution | ✓ | ✓ | Works on both majors after 0.2.1. |
| `download_artifacts(run_id, <name>)` | ✓ | ✓ | Default + custom model names. |
| Run-level `log_artifacts` + download (the plugin's `ario/`) | ✓ | ✓ | Training/registration/prediction artifact witnesses persist on both. |
| `create_model_version(source="runs:/<rid>/<name>")` | ✓ | ✓ | `mv.source` stays `runs:/…` on v3 (not rewritten). |
| `pyfunc.load_model("runs:/<rid>/<name>")` — `VerifiedModel`'s load path | ✓ | ✓ | The plugin loads via `mv.source` (not `models:/<name>/<v>`) precisely so this path stays on the working surface. |
| `pyfunc.load_model("models:/<name>/<version>")` for a `runs:/`-sourced version | ✓ | ✗ | Fails on v3; **the plugin doesn't use this path.** |
| `transition_model_version_stage` | ✓ | ✓ (deprecated) | Promotion works on both. Aliases are the v3+ idiom; stage transitions still functional. |
| `get_model_version` / `set`+`get_registered_model_alias` | ✓ | ✓ | |
| Top-level `mlflow.get_active_trace_id()` | **absent** | ✓ | v3-only top-level helper → fixed in 0.2.2 (fall back to `mlflow.get_current_active_span().request_id` on v2). |
| `mlflow.get_current_active_span()` inside `@mlflow.trace` | ✓ | ✓ | The cross-version anchor for "current trace id." On v2 the id is `span.request_id` (`trace_id` is `None`); on v3 either field has it. |
| Top-level `mlflow.set_trace_tag` | **absent** | ✓ | v3-only top-level helper → fixed in 0.2.2 (use `MlflowClient().set_trace_tag` on both). |
| `MlflowClient.set_trace_tag` | ✓ | ✓ | Cross-version write API; persists mid-span and post-span on both majors when read via `_fetch_trace_tags`. |
| `MlflowClient._tracing_client.get_trace_info` | **absent** | ✓ | Verify's `_fetch_trace_tags` falls back to `client.get_trace` on v2 — already handled. |
| `run.inputs.dataset_inputs` (entity-shaped) | ✓ | ✓ | `_serialize_dataset_inputs` round-trips through `verify_source_of_truth` on both majors. |
| Filesystem tracking store (`./mlruns`, the default) | ✓ | ✓ (deprecated) | Deprecated upstream Feb 2026 → prefer `sqlite:///…`. |
| `search_model_versions("name='…' and current_stage='…'")` (legacy stage URI resolver) | ✓ | **rejected** | v3 dropped `current_stage` from the search grammar → fixed in 0.2.3 (Python-side fallback enumerates versions, filters by `mv.current_stage`). |
| `MlflowClient.get_logged_model` (`models:/<model_id>` v3-native URI) | **absent** | ✓ | The LoggedModel direct URI. Added in 0.2.3 to `_resolve_model_version` so `VerifiedModel` integrity-verifies this form too instead of silently degrading. |
| `mlflow.<flavor>.log_model(..., registered_model_name=…)` auto-register-at-log-time | ✓ | ✓ | Creates a model version for the current run *before* `anchor()` returns — the only flow that produces a usable `_find_registered_model_for_run` lookup, and therefore the only flow where the `ario.last_training_hash` chain head fires. The README's separate-register pattern can't chain training-→-training; the auto-register pattern can. |

## Phases — history

### Phase A — v3 artifact auto-resolution + CI gate · ✅ shipped in 0.2.1
`_logged_model_paths()` reads `run.outputs.model_outputs` on v3; `tests/test_mlflow3_integration.py` + a CI `integration` job run on both majors.

### Phase B — integration coverage for the remaining flows · ✅ shipped in 0.2.2
Each row below was written as a real-MLflow integration test, run on 2.22 + 3.12, with only confirmed failures fixed. **All four are green on both majors.**

| # | Flow | Test | Outcome |
|---|------|------|---------|
| B1 | Registration + promotion (`ArioMlflowClient`) | `test_registration_and_promotion_full_flow` | Green on both. No behavior change. The v3 "create_model_version source handling" risk did not materialize — `mv.source` stays `runs:/…` and `artifact_verified == "true"` end-to-end on both. |
| B2 | `VerifiedModel` end-to-end + tamper rejection | `test_verified_model_predict_full_flow` + `test_verified_model_rejects_tampered_artifact` | Green on both. `IntegrityError` raised before pyfunc load on tampered artifact, on both. |
| B3 | `full_verify` (sig + anchored bytes + live source-of-truth) for training / registration / prediction | `test_verify_full_chain_against_live_mlflow` | Green on both — **after** fixing two real MLflow-2.x bugs the coverage surfaced (see below). |
| B4 | Dataset anchoring (in-training + standalone) | `test_in_training_dataset_anchoring` + `test_standalone_dataset_event_signed_and_verifiable` | Green on both. `_serialize_dataset_inputs` round-trips through source-of-truth on both. |

**Surprise finding — two MLflow-2.x prediction-verification bugs fixed in 0.2.2:**

1. **`mlflow.get_active_trace_id` doesn't exist on 2.x** — `VerifiedModel.predict` swallowed the `AttributeError` → `trace_id = None` → the prediction payload omitted `mlflow_trace_id` → check 3 returned `live_refetch_incomplete`. Fixed via `_active_trace_id()` shim that falls back to `mlflow.get_current_active_span().request_id` on 2.x.
2. **`mlflow.set_trace_tag` doesn't exist on 2.x** — none of the predict-path trace tags (`ario.payload_json`, `ario.decision_id`, etc.) were ever written on 2.x; everything was AttributeError-swallowed. Fixed by routing through `self._mlflow_client.set_trace_tag` (works on both majors).

Both were silently failing despite the 0.1.0 CHANGELOG claiming prediction verification "works on both MLflow 2.x and 3.x" — the discipline lesson is on us, not MLflow.

### Phase C — CI, docs, release · ✅ shipped in 0.2.2
- CI `integration` matrix already runs `tests/test_mlflow3_integration.py` on `mlflow<3` and `mlflow>=3` — picks up B1–B4 automatically.
- Docs (this file + README + CLAUDE.md) flipped to "**MLflow 2.x and 3.x are both fully supported.**"
- Released as `0.2.2` (patch — the only behavior changes were bug fixes for MLflow-2.x predict-side verification that had never worked).

### Post-audit — URI-resolution fixes + edge coverage · ✅ shipped in 0.2.3
A focused post-0.2.2 review probed v3-sensitive edges Phase B had assumed-correct without asserting. Two real `models:/` URI bugs surfaced and were fixed in `_resolve_model_version` (`ario_mlflow/model.py`); the integration suite grew six tests so the next change can't silently regress what 0.2.3 just nailed down:

- **Stage URI on v3** — native `current_stage` search rejected; Python-side fallback added.
- **`models:/<model_id>` on v3** — handled via `client.get_logged_model`; integrity check now runs (previously silently skipped).
- **All four `models:/` URI forms** through `VerifiedModel` get a dedicated test now (numeric, alias, stage, LoggedModel-id v3-only).
- **Multi-model `_logged_model_paths`** and the `anchor()` disambiguation `ValueError` — tested on both majors.
- **Multi-dataset input** — `_serialize_dataset_inputs` round-trips through `verify_source_of_truth` for >1 input.
- **Training-→-training chain** via `log_model(registered_model_name=…)` auto-register — pinned. The README's separate-register pattern can't chain training-→-training; the auto-register idiom can, and is the documented path going forward.

### Ship-readiness pass — boundary CI matrix + mocked-suite cross-version cleanup + live-network smoke · ✅ shipped in 0.2.4
A pre-release audit ("are we sure we're ready to ship to devs?") closed the last four gaps the prior phases left:

- **Mocked unit suite is finally cross-version-clean.** Eleven `monkeypatch.setattr(mlflow, "get_active_trace_id", …)` sites across `tests/test_plugin_smoke.py` and `tests/test_input_anchoring.py` were missing `raising=False`, so they `AttributeError`'d on MLflow 2.x and surfaced as ~22 failures every run. The plugin's own product code already handled the absence via `_active_trace_id()`; only the tests needed the fix. **178/178 mocked tests now pass on real MLflow 2.14, 2.22, 3.0, and 3.12.**
- **Boundary CI matrix.** The integration job now runs against `==2.14.*`, `==2.22.*`, `==3.0.*`, and `==3.12.*` rather than just "latest of each." Same job also runs the mocked suite so the cleanliness above stays asserted.
- **Live-network smoke** (`tests/test_live_network.py`) gated behind `ARIO_MLFLOW_LIVE_NETWORK=1` — covers real Turbo upload → multi-gateway fetch → signature-verify round trip, `verify_proof_by_tx` against a fresh upload, multi-gateway fetch fallback, and the optional ar.io Verify 4th check. Tests skip cleanly when the wallet is unfunded.
- **Manual `workflow_dispatch` workflow** (`.github/workflows/live-network.yml`) so maintainers can trigger the live smoke pre-release with a funded-wallet secret.

### Phase D — future-proofing (deprecations, NOT current breaks) · defer
- **Aliases for promotion** — before MLflow eventually removes stages (still functional + deprecated on 3.12).
- **`get_last_active_trace_id` shim** — if `get_active_trace_id` is ever removed from 3.x.
- **sqlite backend guidance** in README/`plugin-production.md` (file store deprecated Feb 2026).

## Known pre-existing limitation (not a v3 regression)

`anchor(dataset=ds)` reads `ds.name`, `ds.source` (str), `ds.source_type` (str), `ds.digest`, `ds.schema` — the shape of `mlflow.entities.Dataset` (an entity from `run.inputs.dataset_inputs[i].dataset`). A *live* `mlflow.data.from_numpy(...)` object has no `source_type` string and its `source` is a `DatasetSource` object, not a string. The in-training path gets the entity automatically; standalone callers should extract the entity from a run input (or build an entity-shaped object). Surfaced in tests against real MLflow; tracked separately from MLflow-version support.
