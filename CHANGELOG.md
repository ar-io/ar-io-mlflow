# Changelog

All notable changes to `ar-io-mlflow` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Active development. Not yet published to PyPI; install from source via
`pip install -e .`.

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
