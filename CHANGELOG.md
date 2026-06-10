# Changelog

All notable changes to `ar-io-mlflow` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`VerifyStatusClient`** (`ario_mlflow.verify_status_client`) — consumer client
  for the ar-io-agent `GET /v1/verify-status/<asset_id>` endpoint (wire contract:
  `ar-io-agent/docs/verify-status-api.md`, v1.2 Lane A). Supports both deployment
  forms: the same-host management port (`secret=` → `X-Ario-Management-Secret`)
  and the api-guard proxy (`api_key=` → `Authorization: Bearer`). Branches on
  HTTP status codes only, normalizes unrecognized outcomes to `unknown`, ignores
  unknown response fields (contract §10), and offers opt-in monotonic-clock
  response caching for hot-path consumers (contract §9.2).
- **Typed verify-status exception family** (`ario_mlflow.errors`) —
  `AssetVerificationError` (family root), `VerifyStatusError`,
  `AssetTamperedError`, `AssetStaleError`, `AssetMissingError`,
  `AssetUnknownError`, `VerifyStatusAuthError`, `VerifyStatusUnknownAssetError`,
  `VerifyStatusTransportError`, and `VerifyStatusLicenseError` (api-guard's
  `503 license required`, carrying `upgrade_url`). The §9.1 outcome→exception
  mapping lives in `errors.exception_for_status`.
- **`VerifiedModel` agent verify-status gate** — new optional keyword-only
  arguments `asset_id=`, `verify_status_client=`, `on_failure=` on
  `VerifiedModel.__init__`. When provided, the agent gate runs **first** (before
  the artifact integrity check and before any MLflow access — fail fast, one
  cheap local HTTP call) and refuses to load a model whose covering asset is
  tampered/missing/stale/unknown per the contract §9.1 mapping. `on_failure`:
  `"raise"` (default) and `"fail_closed"` raise the typed exception;
  `"fail_open"` logs at WARN with structured SIEM fields
  (`extra["ario_verify_status"]`: asset_id, outcome, stale, policy_hash,
  current_tx_id) and proceeds. Absent the new kwargs, behavior is byte-for-byte
  unchanged. Gating is load-time only in this release; per-predict re-checking
  (contract §9.2's 10–30s cache cadence) is a planned follow-up.

### Changed

- `IntegrityError` now subclasses `ario_mlflow.errors.AssetVerificationError`
  (previously bare `Exception`; backward-compatible) so one
  `except AssetVerificationError` clause catches both load-time gates —
  artifact re-hash and agent verify-status.

## [0.2.4] — 2026-05-28

Ship-readiness pass: makes the mocked unit suite cross-version-clean, broadens the
real-MLflow CI matrix to the **boundary** versions of each major (not just latest),
and adds an opt-in **live-network smoke** suite + manual GitHub Actions workflow so
maintainers have an end-to-end path that exercises real Arweave uploads, gateway
fetches, and the optional ar.io Verify 4th check before each release.

### Fixed

- **22 mocked unit-test failures on MLflow 2.x.** Eleven
  ``monkeypatch.setattr(mlflow, "get_active_trace_id", …)`` sites (across
  `tests/test_plugin_smoke.py` and `tests/test_input_anchoring.py`) didn't pass
  ``raising=False``, so they raised `AttributeError` on MLflow 2.x where the
  top-level helper is absent — every test that touched the predict path failed.
  All sites now pass `raising=False`; the plugin's existing
  `_active_trace_id()` shim handles the fallback in the actual product code.
  **Result: 178/178 mocked tests now pass on real MLflow 2.14, 2.22, 3.0, and
  3.12** — the mocked suite is finally cross-version-clean instead of only
  cross-version-assumed.

### Documentation

- **README install instructions no longer claim "PyPI publish is on the
  roadmap"** — `pip install ar-io-mlflow` has worked since 0.2.0 and is now
  the documented primary path. Added the MLflow boundary-version list.
- **`docs/mlflow-v3-support.md` status line** updated to reflect the boundary
  CI matrix (2.14 / 2.22 / 3.0 / 3.12) rather than just 2.22 / 3.12.
- **`docs/solana-wallet-plan.md` marked HISTORICAL** — that doc was the design
  plan for 0.2.0 (Solana wallet support); kept for reference but new work
  should be specced against the live code + CHANGELOG, not the plan doc.
- **CHANGELOG `[0.1.0]` "Not yet published to PyPI" wording corrected** to
  "(Pre-PyPI; 0.2.0 was the first published release.)"
- **Threat model + README** clarified that semantic-verification is *out of
  scope by design*, not "on the roadmap, not in v0.1" (deferred-feature
  framing that hadn't been true for several releases).

### Added

- **CI integration matrix now covers the boundary MLflow versions of each major**
  (`==2.14.*`, `==2.22.*`, `==3.0.*`, `==3.12.*`) rather than just "latest each."
  Catches regressions on the pyproject floor (2.14) and the first 3.x release
  (3.0, when LoggedModel became first-class) before users hit them. The same job
  also runs the mocked suite on each version so the cross-version cleanliness
  established above stays asserted, not assumed. (2.14 needs `setuptools<80` to
  keep `pkg_resources` importable; pinned in the install step.)
- **Live-network smoke test suite** (`tests/test_live_network.py`) gated behind
  `ARIO_MLFLOW_LIVE_NETWORK=1`. Covers the path the rest of the suite can't:
  - wallet construction against the production resolution rules,
  - real Turbo upload → multi-gateway fetch → signature-verify round trip,
  - `verify_proof_by_tx` against a freshly anchored TX,
  - multi-gateway fetch fallback when the primary is unreachable,
  - the optional ar.io Verify 4th check (gated additionally on
    `ARIO_MLFLOW_ARIO_VERIFY_URL`).

  Tests skip cleanly (not fail) when the wallet is unfunded — auto-generated
  Solana wallets surface "insufficient credits" via `anchor.last_error` and the
  suite distinguishes that from a real upload failure.
- **Manual `workflow_dispatch` CI workflow** (`.github/workflows/live-network.yml`)
  that maintainers can trigger pre-release. Accepts the wallet contents via a
  repository secret (`ARIO_MLFLOW_LIVE_WALLET_JSON`, Solana id.json array or
  Arweave RSA JWK) and an optional ar.io Verify URL secret
  (`ARIO_MLFLOW_LIVE_ARIO_VERIFY_URL`). Without the wallet secret the funded tests
  skip with an actionable message rather than failing.

## [0.2.3] — 2026-05-28

A targeted post-0.2.2 audit found that the v3 Phase B coverage missed
two real `models:/` URI bugs and several edges the plan had flagged but
never asserted. This release closes both.

### Fixed

- **`VerifiedModel("models:/<name>/<stage>")` on MLflow 3.x.** MLflow 3
  dropped `current_stage` from `search_model_versions`'s valid attribute
  grammar, so the resolver's native stage search raised
  `MlflowException: Invalid attribute key 'current_stage'`, was swallowed,
  and `VerifiedModel` then tried to `pyfunc.load_model("models:/<name>/<stage>")`
  directly — which fails on v3 with "no MLmodel". `_resolve_model_version`
  now falls back to `search_model_versions(name='<name>')` and filters by
  `mv.current_stage` in Python, so v2 codebases that load by legacy stage URI
  upgrade to MLflow 3 cleanly. Aliases remain the v3-native idiom; the
  fallback is the bridge.
- **`VerifiedModel("models:/<model_id>")` (v3-native LoggedModel URI).**
  Before this fix, the resolver returned `None` for the single-segment v3
  URI form, `VerifiedModel` silently skipped the integrity check
  (`_artifact_verified=None`) and predictions chained at `GENESIS` instead
  of the registration. The user got a working `VerifiedModel` with **no
  integrity guarantee** — the worst kind of regression. `_resolve_model_version`
  now calls `client.get_logged_model(<model_id>)` on v3 and synthesizes a
  ModelVersion-shaped handle (with `run_id` / `source` / `name` / `version`)
  so the integrity-check and predict paths run as on every other URI form.

### Added — integration coverage for the edges Phase B left implicit

`tests/test_mlflow3_integration.py` now also exercises (on both majors
where applicable):

- All four `models:/` URI forms through `VerifiedModel` — numeric, alias,
  legacy stage (Python-side fallback on v3), and v3-native `models:/<model_id>`
  (v3-gated). Each must produce `artifact_verified=True` end to end.
- Multi-model-output guard — `_logged_model_paths()` enumerates every
  model logged in a run on v3 via `run.outputs.model_outputs`, and
  `anchor()` raises a clear `ValueError` when no `artifact_path` is
  given, then succeeds with an explicit one.
- Multi-dataset input — `_serialize_dataset_inputs` produces a stable
  list of >1 dataset entries that round-trips identically through
  `verify_source_of_truth` on both majors.
- **Training-→-training chain** via the auto-register-at-log-time idiom
  (`mlflow.<flavor>.log_model(name=..., registered_model_name=...)`).
  This is the only realistic flow that produces a model version *before*
  `anchor()` runs, which is what `_find_registered_model_for_run` needs
  to write `ario.last_training_hash`. The README's separate-register
  pattern can't chain training-→-training because no version exists at
  anchor time; the auto-register pattern does, and this test pins it.
  Documented in [`docs/mlflow-v3-support.md`](docs/mlflow-v3-support.md).

### Documentation

- `docs/mlflow-v3-support.md` updated with the two URI fixes, the audit
  findings, and an explicit note that the training-→-training chain
  requires the auto-register-at-log-time idiom (not the separate-register
  pattern in older README examples). The verified behavior matrix now
  includes rows for stage-search v3 attribute grammar and
  `client.get_logged_model`.

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
resilience pass. (Pre-PyPI; 0.2.0 was the first published release.)

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
