# Changelog

All notable changes to `ar-io-mlflow` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No changes yet._

## [0.2.2] — 2026-05-28

### Fixed

- **MLflow 2.x prediction-side trace correlation and verification.** Two
  top-level MLflow APIs that the predict path used are 3.x-only and silently
  raised `AttributeError` on 2.x, leaving prediction proofs without a usable
  trace id and prediction source-of-truth verification permanently failing on
  2.x despite the 0.1.0 changelog claiming the path "works on both 2.x and
  3.x". Both surfaced via the new Phase B integration coverage on real
  MLflow 2.22.

  - **`mlflow.get_active_trace_id` is 3.x-only.** `VerifiedModel.predict`
    now resolves the active trace id via a cross-version `_active_trace_id()`
    shim that falls back to `mlflow.get_current_active_span().request_id` on
    2.x (where `trace_id` is `None` and the id lives on `request_id`).
    Without this, prediction payloads on 2.x omitted `mlflow_trace_id` and
    `verify_source_of_truth` returned `live_refetch_incomplete`.
  - **`mlflow.set_trace_tag` is 3.x-only too.** All prediction-path trace
    tags (`ario.payload_json`, `ario.decision_id`, `ario.input_hash`,
    `ario.output_hash`, `ario.payload_hash`, `ario.proof_status`,
    `ario.prediction_tx`, etc.) now route through
    `MlflowClient.set_trace_tag` (works on both majors) via a new
    `VerifiedModel._mlflow_client`. Without this, none of these tags were
    ever written on 2.x, so `ario.payload_json` was missing for verify and
    check 3 always failed.

### Added

- **MLflow 2.x + 3.x integration coverage across every plugin flow.**
  `tests/test_mlflow3_integration.py` now exercises the full plugin surface
  on real MLflow tracking stores: `ArioMlflowClient` registration + promotion
  with chain linkage (B1), `VerifiedModel` load-time integrity + per-prediction
  anchoring + tamper rejection (B2), `full_verify` against live MLflow for
  training / registration / prediction (B3, which surfaced the two fixes
  above), and dataset anchoring — in-training and standalone (B4). The CI
  `integration` matrix runs the file on `mlflow<3` and `mlflow>=3` so MLflow
  version support stays tested rather than assumed.

### Changed

- **Docs: "fully supported" on both majors.** `README.md`, `CLAUDE.md`, and
  `docs/mlflow-v3-support.md` now state plainly that MLflow 2.x (2.14+) and
  3.x are both fully supported for every plugin flow, with the verified
  behavior matrix and the cross-version API differences the plugin handles
  documented in one place.

## [0.2.1] — 2026-05-27

### Fixed

- **MLflow 3.x model artifact resolution.** MLflow 3 makes models first-class
  `LoggedModel` entities and drops the `mlflow.log-model.history` run tag, so
  the plugin's artifact-path auto-resolution returned nothing on v3 — a model
  logged under a non-default name fell back to `"model"` and had its artifact
  hash **silently skipped**. `_logged_model_paths()` now reads
  `run.outputs.model_outputs` (→ `get_logged_model(model_id).name`) on v3,
  falling back to the tag on v2, restoring artifact-hash integrity for the
  `anchor()` and verify paths. Verified on real MLflow 2.22 and 3.12.

### Changed

- **Honest MLflow version-compatibility docs.** No longer claims blanket
  "tested 2.14 through 3.x." MLflow 2.x and 3.x are supported for the core
  anchor/verify (artifact-hashing) path, now tested in CI against **both**
  majors via a real-MLflow integration test (`tests/test_mlflow3_integration.py`).
  Registration/promotion, prediction, and full `verify_record` still need
  dedicated v3 integration coverage (tracked in `docs/mlflow-v3-support.md`).
  Note: MLflow's filesystem tracking store (the default) is deprecated upstream
  as of Feb 2026 — prefer a `sqlite:///…` backend for new setups.

## [0.2.0] — 2026-05-27

The first release beyond the initial alpha — Solana-default funding wallet,
the `spec_version` envelope field + cross-tool verification, a
machine-readable audit bundle, and a full programmatic verification
reference. Install from source via `pip install -e .` (PyPI publish
pending).

### Added

- **Solana (ed25519) is the default upload/funding wallet** — newly
  generated wallets are now Solana keypairs (persisted as a Solana CLI
  `id.json` at the existing `~/.ario-mlflow/wallet.json` path), uploaded to
  Arweave via Turbo `SolanaSigner` (ANS-104 sigType 2). **Arweave RSA still
  works and is selectable.** The chain is detected from the key's JSON
  shape — an RSA JWK object → `arweave`, a 64-int `id.json` array or base58
  string → `solana` — so there is still **one** env var
  (`ARIO_MLFLOW_ARWEAVE_WALLET`) and **one** wallet path, both accepting
  either chain. A pre-existing RSA `wallet.json` is detected and reused
  unchanged (never overwritten); zero config required for the Solana
  default. `WalletLoadError` discipline is preserved for both chains (a
  malformed caller-supplied wallet never silently substitutes). The chain
  is surfaced in logs, the `ario/verification.html` wallet banner, and the
  new `ario.wallet_type` MLflow run tag. Upload/funding layer only — the
  Ed25519 envelope-signing key (`ProofEngine`), the envelope/proof format,
  and `verify_record` are unchanged.
- **`ar-io-mlflow audit <model> --format=json [--output <path>]`** — the
  model-lineage audit now emits a machine-readable evidence bundle
  (`ario.mlflow.audit/v1`), parallel to the agent's `ariod audit export`.
  Brings the two repos to reporting parity: a compliance team can now pull
  a structured lineage artifact (training → registration → promotion →
  artifact integrity, each with per-check results) for SOC2/ISO27001
  evidence, not just a terminal screenshot. JSON mode is pipe-clean (no
  panel rendered to stdout); `--output` writes to a file. Default remains
  the human-readable terminal panel — `text` mode is byte-unchanged.
- **`spec_version` field on signed envelopes** — every envelope minted by
  `create_commitment()` now carries `spec_version: "ario.mlflow/v1"` as
  part of the signed body. Pins the envelope shape so a future spec bump
  can roll out via the same field. Exported as `ario_mlflow.SPEC_VERSION`.
- **Cross-tool spec acceptance** — `verify_commitment()` and
  `verify_signature()` accept any envelope whose `spec_version` major is
  in the recognized set, not only plugin-minted `ario.mlflow/v1`.
  Envelopes produced by any other tool that shares the spec (RFC-8785 JCS
  + SHA-256 + Ed25519) verify under the plugin's verifier.

### Changed

- **`turbo-sdk>=0.1.0`** (was `>=0.0.5`) — required for `SolanaSigner`.
- **Verifier results expose `spec_version_status` and `legacy_envelope`**
  — `ProofEngine.verify_commitment()` and `verify.verify_signature()`
  now report whether the envelope's `spec_version` is `"supported"`,
  `"legacy"` (field absent — envelopes anchored before this build
  continue to verify normally), or `"unsupported"` (unknown major —
  `overall` is `False` with `reason="unsupported_spec_version"`).

### Documentation

- **New [`docs/verification.md`](docs/verification.md)** — full
  verification reference: the four checks and their tri-state (`True` /
  `False` / `None`) result semantics, the programmatic API
  (`verify_record` / `verify_proof_by_tx` / `full_verify`) with a
  when-to-use table and examples, per-event-type nuances, the
  `ario/verification.html` report, `spec_version` / cross-product
  handling, and a copy-paste CI/monitoring job.
- **README "Programmatic verification" section** — documents the three
  composite verify functions, the tri-state result contract (incl. the
  "required MLflow checks fail `overall` when no client is supplied"
  gotcha), and agent cross-product verification. The "Verifying without
  Python" auditor recipe now includes a `spec_version` check.

## [0.1.0] — 2026

Initial alpha — covers the three integration points, dataset anchoring, the
CLI verify/audit flow, the safety-and-packaging pass, and the network
resilience pass. Not yet published to PyPI.

### Added — core API

- **`anchor()`** — training provenance helper that signs a pure-commitment
  envelope over the active MLflow run's params, metrics, and artifact
  checksums, and uploads it to Arweave via Turbo.
- **`ArioMlflowClient`** — drop-in replacement for `MlflowClient` that
  auto-anchors `create_model_version()` and
  `transition_model_version_stage()` in a daemon thread. Exposes
  `anchor_status()` and `wait_for_anchor()` for callers that need the
  outcome.
- **`VerifiedModel`** — inference wrapper with load-time integrity check
  (raises `IntegrityError` on artifact-hash mismatch before user code runs)
  and per-prediction anchoring in a background thread.
- **Standalone dataset anchoring** — `anchor(dataset=ds)` mints an
  independent signed event with its own Arweave TX, no MLflow run required.
  In-training calls also auto-anchor each logged dataset and reference its
  TX in the training proof.
- **CLI** — `ar-io-mlflow verify run|model|trace <id>` and
  `ar-io-mlflow audit <name>/<version>` for after-the-fact verification and
  full-lineage audits.
- **MLflow `RunContextProvider` entry point** — importing the package
  auto-tags every run with `ario.enabled` and `ario.version`.
- **OpenTelemetry correlation** — auto-captures `otel_trace_id` /
  `otel_span_id` into the canonical payload when an active span exists, so
  proofs are correlatable with infrastructure tracing.
- **HTML verification report** generated as an MLflow artifact
  (`ario/verification.html`) on each anchored event.

### Added — safety pass

- **`WalletLoadError`** — raised from `ArweaveAnchor(wallet_path=...)` when
  a caller-supplied wallet path is missing or malformed. Replaces silent
  fallback to an auto-generated wallet, which would have signed proofs
  under a different on-chain identity with no programmatic signal.
- **PEP 621 packaging** — migrated from `setup.py` to `pyproject.toml` with
  full PyPI metadata, classifiers, and `__version__` exposed via
  `ario_mlflow.__version__`.
- **Apache-2.0 LICENSE** at repo root, matching ar.io org convention.

### Added — resilience pass

- **HTTP retry with exponential backoff** — `ArweaveAnchor` and
  `ArioVerifyClient` share a `requests.Session` with a `urllib3` Retry
  adapter. 5xx and 429 responses retry with exponential backoff, honoring
  `Retry-After`. Configurable via `max_retries` / `retry_backoff_factor`
  constructor kwargs.
- **Multi-gateway fetch fallback** — `ArweaveAnchor.fetch_proof()` walks an
  ordered gateway list (default `["turbo-gateway.com", "ardrive.net"]`,
  override via `gateways=` kwarg or `ARIO_MLFLOW_GATEWAYS` env var) so a
  single flaky gateway no longer surfaces as a hard verify failure.
- **`last_error` introspection** — `ArweaveAnchor` and `ArioVerifyClient`
  expose a `last_error` string attribute populated when methods return
  `None`, so callers can distinguish "anchor disabled" from "retries
  exhausted" without parsing logs.
- **`ArioVerifyClient.poll_attestation()`** — wait for an attestation to
  reach a target maturity level (1 → 2 → 3) with configurable timeout +
  interval. Returns the latest result either way.

### Removed

- **`https://vilenarios.com/local/verify` fallback in `report.py`** — when
  no verify URL is configured, the CLI command stands alone in generated
  HTML reports instead of pointing at a personal endpoint.

### Fixed

- **MLflow 3.x prediction verification** — the prediction-side
  `verify_source_of_truth` previously returned `live_refetch_incomplete`
  on MLflow 3.x because `client.get_trace()` requires
  `mlflow.artifactLocation` in trace tags to load spans, and MLflow 3.x
  doesn't always set that tag. The refetcher now uses
  `_tracing_client.get_trace_info()` (tags-only, no spans load), which
  works on both MLflow 2.x and 3.x. Training and registration
  verification were never affected.

### Known limitations

(none currently tracked — open an issue if you hit one)
