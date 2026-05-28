# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`ar-io-mlflow` is an MLflow plugin (PyPI: `ar-io-mlflow`, import: `ario_mlflow`) that adds verifiable provenance to the MLflow lifecycle. It signs ~500-byte commitments for training, registration, promotion, and prediction events, then anchors them to Arweave via the ar.io Turbo bundler. Source data (params, metrics, artifacts, inputs, outputs) never leaves MLflow — only the SHA-256 commitment goes on chain.

`mlflow.db` and `mlruns/` in the repo root are local artifacts from manual experimentation, not canonical state — ignore them when reasoning about the project.

## Common commands

```bash
# Install editable with test extras (required before running tests)
pip install -e ".[test]"

# Tests — no network or MLflow server required
pytest                                              # full suite
pytest tests/test_plugin_smoke.py                   # one file
pytest tests/test_plugin_smoke.py::test_name        # one test
pytest -k wallet                                    # by keyword
pytest -v --tb=short                                # what CI runs (.github/workflows/test.yml)

# CLI (installed as console script by pyproject.toml)
ar-io-mlflow verify run <run_id>
ar-io-mlflow verify model <name>/<version>
ar-io-mlflow verify trace <trace_id>
ar-io-mlflow audit <name>/<version>                       # human-readable terminal panel
ar-io-mlflow audit <name>/<version> --format=json         # machine-readable bundle to stdout
ar-io-mlflow audit <name>/<version> --format=json --output bundle.json
```

The CLI reads `MLFLOW_TRACKING_URI` (default `./mlruns`). Export it to match the store used at training time, or the run lookup fails.

`audit --format=json` emits an `ario.mlflow.audit/v1` evidence bundle (model + version + per-stage tx/checks/ok + artifact_hash + overall_ok), the lineage parallel to the agent's `ariod audit export`. JSON mode suppresses the terminal panel so stdout is pipe-clean; `--output` writes to a file. `text` mode (default) is unchanged.

There is no separate lint/format step configured in this repo.

## High-level architecture

### The core invariant

**Arweave is a witness; MLflow is the system of record.** The plugin does not put MLflow data on Arweave. It commits a hash so anyone can verify "what's in MLflow now matches what was anchored at time T." Auditor independence is non-negotiable: the proof spec (RFC-8785 JCS + SHA-256 + Ed25519) is reproducible in any language without this plugin.

### Three public integration points (`ario_mlflow/__init__.py`)

1. **`anchor()`** (`anchoring.py`) — call inside `mlflow.start_run()` after `log_model()`. **Synchronous**: hashes artifacts, signs the envelope, uploads to Turbo, writes `ario.*` tags. Returns a dict with `envelope`, `payload_hash`, `tags`, `artifact_status`, etc. Raises `ArtifactAccessError` on hashing failure, `WalletLoadError` on bad caller-supplied wallet, `RuntimeError` with no active run.
2. **`ArioMlflowClient`** (`client.py`) — drop-in `MlflowClient` subclass. `create_model_version()` and `transition_model_version_stage()` return immediately; anchoring runs in a **daemon thread**. Observe via `anchor_status(event_type, name, version)` and `wait_for_anchor(...)`. Anchoring failures never break the underlying MLflow call.
3. **`VerifiedModel`** (`model.py`) — wraps `models:/...` URI. `__init__` is **synchronous** and re-hashes artifacts against `ario.artifact_hash`, raising `IntegrityError` *before* loading user pyfunc code. `predict()` returns immediately; per-prediction anchoring runs in a daemon thread. Each result exposes `proof_status` / `tx_id` / `wait_for_anchor()`.

Plus standalone dataset anchoring: `anchor(dataset=ds)` mints an independent signed event with no MLflow run required.

### Per-event-type chain semantics (DAG, not a single chain)

Events form a DAG because MLflow has no compare-and-set primitive. Each event type chains independently:

- **Training** chains via the registered model's `ario.last_training_hash` tag (read for `previous_hash`, written with the new payload hash).
- **Registration** chains to the source training run's `ario.training_tx` tag.
- **Prediction** chains to the model version's `ario.registration_tx` tag, **read once at `VerifiedModel.__init__`**. Predictions never re-read or write the chain head — this sidesteps races on the high-frequency busy path. If registration hasn't completed yet, early predictions chain to `GENESIS`.

When editing chain logic, preserve this asymmetry. Writing the chain head on the predict path is a regression.

### Module layout (`ario_mlflow/`)

- `proof.py` — Ed25519 sign/verify, RFC-8785 (JCS) canonicalization via the `jcs` package, SHA-256. The cryptographic core; everything else depends on this.
- `anchoring.py` — `anchor()` entry point, artifact-checksum helpers, OTel auto-capture (`ARIO_MLFLOW_CAPTURE_OTEL` opt-out), the `ario.last_training_hash` tag constant, `ArtifactAccessError`.
- `arweave.py` — `ArweaveAnchor` (Turbo uploads, multi-gateway fetch, `requests.Session` + urllib3 Retry adapter). `WalletLoadError` lives here. Default wallet at `~/.ario-mlflow/wallet.json`. Default fetch gateways: `["turbo-gateway.com", "ardrive.net"]`, override via `ARIO_MLFLOW_GATEWAYS`.
- `client.py` — `ArioMlflowClient`, daemon-thread anchoring, per-`(event_type, name, version)` status tracking with a `threading.Lock`.
- `model.py` — `VerifiedModel`, `IntegrityError`, pyfunc loading guarded by an artifact re-hash.
- `verify.py` — the four verification checks (`verify_signature`, `verify_anchored_bytes`, `verify_source_of_truth`, `verify_ario_attestation`) plus the `full_verify` composite, the auditor-shaped `verify_record`, and operator-side `verify_proof_by_tx`. `ArioVerifyClient` (ar.io Verify REST client with `poll_attestation`) also lives here.
- `cli.py` — `ar-io-mlflow` console-script entry. Renders the three-row verify panel ("Proof Found / Record Matches / Signature Confirmed") plus optional ar.io attestation. Honors `NO_COLOR`.
- `report.py` — generates `ario/verification.html` as an MLflow artifact on each anchored event.
- `plugin.py` — MLflow `RunContextProvider` registered via the `mlflow.run_context_provider` entry point in `pyproject.toml`. Importing the package auto-tags every run with `ario.enabled` / `ario.version`.

### Verification flow (CLI and library)

Every verify surface runs the same three checks plus an optional fourth:

1. **Proof Found** — fetch the envelope from ar.io for the recorded TX ID.
2. **{Event} Record Matches** — download `ario/payload.json` from MLflow, re-hash, compare to `payload_hash`, *and* re-derive the canonical bytes from a *separate* MLflow surface (run params/metrics for training, the `ario.payload_json` trace tag for predictions). Both must agree — catches post-anchoring MLflow tampering.
3. **Signature Confirmed** — Ed25519 verify of the envelope's signature against its embedded `public_key`.
4. **ar.io attestation** *(optional, when `ARIO_MLFLOW_ARIO_VERIFY_URL` is set)* — independent gateway-operator check.

Internal field names (`signature_valid`, `hash_match`, `source_of_truth_ok`, `attestation_level`, `permanent_copy_found`) are stable API — only the printed labels follow the dashboard vocabulary.

### MLflow version compatibility

**MLflow 2.x (2.14+) and 3.x are both fully supported** — every plugin flow (training anchor, `ArioMlflowClient` registration/promotion, `VerifiedModel` load+predict, full `verify` with live source-of-truth refetch, dataset anchoring) is integration-tested on **real MLflow 2.22 and 3.12** via `tests/test_mlflow3_integration.py` (gated behind `ARIO_MLFLOW_INTEGRATION=1`; CI runs it on both majors). See [`docs/mlflow-v3-support.md`](docs/mlflow-v3-support.md).

**The real cross-version differences the plugin handles** (do not regress these):

- **v3 — dropped `mlflow.log-model.history` run tag.** Models are first-class `LoggedModel` entities in v3; `_logged_model_paths()` reads `run.outputs.model_outputs` → `get_logged_model(model_id).name` on v3, falling back to the tag on v2. Before this fix, a custom-named model on v3 fell back to `"model"` and its hash was silently skipped — feeds both `anchor()` and the verify-side refetcher.
- **v2 — `mlflow.get_active_trace_id` doesn't exist there.** It's a 3.x top-level helper. The `_active_trace_id()` shim in `model.py` falls back to `mlflow.get_current_active_span().request_id` on v2 (where `trace_id` is `None` and the id lives on `request_id`). Without this, predict-side trace correlation silently produced `None` on v2, breaking prediction source-of-truth verification.
- **v2 — `mlflow.set_trace_tag` doesn't exist there either.** Use `MlflowClient().set_trace_tag` (works on both). `VerifiedModel` keeps a `self._mlflow_client` for this. Top-level `mlflow.set_trace_tag` is 3.x-only and was the silent cause of "no `ario.payload_json` trace tag → prediction verify check 3 always fails" on v2.
- **v3 — trace info accessor moved.** `MlflowClient._tracing_client.get_trace_info` (tags-only, sidesteps v3's `mlflow.artifactLocation` requirement on `get_trace`) is preferred; `_fetch_trace_tags` in `verify.py` falls back to `client.get_trace` on v2 where the private attribute doesn't exist.

**Deprecation to watch:** the filesystem tracking backend (`./mlruns`, the plugin/CLI default) is deprecated upstream as of Feb 2026 — prefer a `sqlite:///…` backend for new setups.

**Known pre-existing limitation (not a v3 regression):** `anchor(dataset=ds)` expects an `mlflow.entities.Dataset` (entity-shaped: string `source_type`/`source`/`schema`), not a live `mlflow.data.from_numpy(...)` object whose `source` is a `DatasetSource` and which has no `source_type` string. The in-training path (via `run.inputs.dataset_inputs`) gives you the entity shape automatically. Standalone callers should extract the entity from a run input, or build an entity-shaped object.

## Conventions to preserve

- **Public API surface** is the re-exports in `ario_mlflow/__init__.py` (lazy-loaded via `__getattr__`). Keep new public symbols there.
- **Version lives in two places** that must stay in sync: `pyproject.toml` `version` and `ario_mlflow/__init__.py` `__version__`. `tests/test_plugin_safety.py::test_version_matches_pyproject_toml` enforces this — bump both together when releasing.
- **`ario.*` tag namespace** is the contract with downstream verifiers and the demo app. Don't rename existing tags. See `README.md` "Tags the plugin writes" for the full list.
- **Sync vs async behavior** is part of the API contract:
  - `anchor()` and `VerifiedModel.__init__` are synchronous (block until complete).
  - `ArioMlflowClient.create_model_version` / `transition_model_version_stage` and `VerifiedModel.predict` return immediately and anchor in daemon threads.
  - Don't change this without updating the README's Performance table.
- **`last_error` introspection** — when `ArweaveAnchor` or `ArioVerifyClient` methods return `None`, the instance's `last_error` attribute carries the cause. Preserve this when adding new failure paths so callers can distinguish "disabled" from "all retries exhausted."
- **Tests use `tmp_path` + `monkeypatch`** for filesystem and env isolation. No network. No real MLflow server. New tests follow this pattern.
- **CHANGELOG entries** under `[Unreleased]` for any user-visible change. PR scope: one cohesive theme per PR.

## Environment variables the plugin reads

| Var | Purpose | Default |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | MLflow store the CLI/verifier reads from | `./mlruns` |
| `ARIO_MLFLOW_ARWEAVE_WALLET` | Path to a funding wallet — Solana `id.json`/base58 **or** Arweave RSA JWK (chain auto-detected by shape); auto-generates a **Solana** wallet under `~/.ario-mlflow/wallet.json` if unset. Existing wallet files are reused, never overwritten | (auto) |
| `ARIO_MLFLOW_GATEWAY_HOST` | Turbo upload host | `turbo-gateway.com` |
| `ARIO_MLFLOW_GATEWAYS` | Comma-separated fetch fallbacks | `turbo-gateway.com,ardrive.net` |
| `ARIO_MLFLOW_SIGNING_KEY` | Ed25519 signing key (base64); generated if unset | (auto) |
| `ARIO_MLFLOW_CAPTURE_OTEL` | Auto-capture OTel into payload; set `false`/`0` to opt out | on |
| `ARIO_MLFLOW_ARIO_VERIFY_URL` | ar.io Verify base URL; enables the 4th check when set | unset |

## Network endpoints the plugin reaches

- `turbo-gateway.com` — Turbo uploads + proof fetches (primary)
- `arweave.net` / `ardrive.net` — fetch fallbacks
- `turbo.ardrive.io` — bundler-status checks
- Configured `ARIO_MLFLOW_ARIO_VERIFY_URL` — optional ar.io attestation

In restricted-egress environments these must be allowlisted. Override the primary host via `ARIO_MLFLOW_GATEWAY_HOST`.

## Further reading in-repo

- `README.md` — full API reference, failure modes, env vars, auditor recipe.
- `docs/architecture.md` — pure-commitment design, JCS canonicalization, evidence chain.
- `docs/plugin-production.md` — wallet ops, CI patterns, monitoring.
- `docs/plugin-threat-model.md` — what the plugin defends against and what it doesn't.
- `CONTRIBUTING.md` — release process, test patterns, scope boundaries.
- `examples/sklearn-quickstart/` — runnable end-to-end example: train → anchor → verify.
