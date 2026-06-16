# ar-io-mlflow

Verifiable provenance for the MLflow lifecycle — training, registration, promotion, inference.
Signed cryptographic proofs are anchored to ar.io, so an auditor can verify a model
or decision long after your MLflow server is gone.

> **Status.** Alpha. The cryptography, packaging, and verification flow are
> stable; default behaviors prioritize frictionless evaluation over production
> hardening. See [`docs/plugin-production.md`](docs/plugin-production.md) for
> deployment guidance and [`CHANGELOG.md`](CHANGELOG.md) for what's shipped.

## Install

```bash
pip install ar-io-mlflow
```

Or, to track main:

```bash
git clone https://github.com/ar-io/ar-io-mlflow.git
cd ar-io-mlflow
pip install -e .
```

Python 3.10+. MLflow 2.14+ and 3.x both supported (boundary versions 2.14, 2.22, 3.0, 3.12 run in CI). Pulls in MLflow, PyNaCl, the ar.io Turbo SDK, `cryptography`, and the shared [`ar-io-proof`](https://pypi.org/project/ar-io-proof/) verification kernel (`>=0.2.0` — the byte-level primitives: JCS canonicalization, SHA-256, Ed25519 sign/verify, the spec-version registry).

### MLflow version compatibility

**MLflow 2.x (2.14+) and 3.x are both fully supported** — every plugin flow
(training anchor, `ArioMlflowClient` registration/promotion, `VerifiedModel`
load + predict, `verify_record` with live source-of-truth refetch, dataset
anchoring) is integration-tested on real MLflow 2.22 and 3.12; CI runs the
gate on both majors. The plugin handles the API surface differences both
ways: v3-only changes (the dropped `mlflow.log-model.history` tag → models
resolved via `run.outputs.model_outputs`; the lighter
`_tracing_client.get_trace_info` for trace-tag refetch) **and** v2-only
patterns (top-level `mlflow.get_active_trace_id` / `mlflow.set_trace_tag`
don't exist on 2.x — the predict path falls back to the active span's
`request_id` and writes trace tags via `MlflowClient.set_trace_tag`).

Note: MLflow's filesystem tracking store (`./mlruns`, the default) is
deprecated upstream as of Feb 2026 — prefer a `sqlite:///…` backend for new
setups. See [`docs/mlflow-v3-support.md`](docs/mlflow-v3-support.md) for the
verified behavior matrix and the v3 support history.

## Quickstart

```python
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import ario_mlflow

# Point MLflow at a tracking store. Skip if MLFLOW_TRACKING_URI is
# already set in your env, or if you're happy with the cwd's ./mlruns.
mlflow.set_tracking_uri("file:///tmp/mlruns")

X, y = load_iris(return_X_y=True)

with mlflow.start_run():
    model = LogisticRegression(max_iter=200).fit(X, y)
    mlflow.log_metric("accuracy", model.score(X, y))
    mlflow.sklearn.log_model(model, name="model")

    # Signs a proof, hashes the logged artifacts, writes ario.* tags,
    # and uploads ~500 bytes to Arweave via Turbo (free for small payloads).
    # allow_empty_dataset_inputs=True opts out of dataset anchoring; see
    # "Dataset anchoring" below for the recommended pattern.
    result = ario_mlflow.anchor(allow_empty_dataset_inputs=True)
    print(result["tags"]["ario.training_tx"])
```

No wallet configured? The plugin auto-generates one on first run and persists it
to `~/.ario-mlflow/wallet.json` so your signing address stays stable across
sessions. Set `ARIO_MLFLOW_ARWEAVE_WALLET=/path/to/wallet.json` to use your own.
The auto-generated wallet starts unfunded — that's fine for typical usage
because Turbo's free tier covers small uploads (see "Wallet & cost" below).

A full runnable example lives in `examples/sklearn-quickstart/`.

## The three integration points

### 1. `ario_mlflow.anchor()` — training provenance

Call inside an active `mlflow.start_run()` after logging your model. The plugin
auto-resolves the logged model's `artifact_path` from MLflow's log-model history,
so you rarely need to pass it explicitly.

Returns a dict with `envelope`, `payload`, `payload_bytes`, `payload_hash`,
`previous_hash`, `anchor_result`, `tags`, `artifact_path`, `artifact_status`
(`"hashed"` / `"no_artifacts"` / `"hash_failed"`), and `artifact_error`.

**Failure modes.** `anchor()` is synchronous and runs to completion before the
`with` block exits.

- **Arweave upload fails** (gateway down, network): the envelope is still
  signed locally and `ario.verify_status` is set to `signed`; `ario.training_tx`
  is absent. Your MLflow run still succeeds. Re-run later to retry. The
  underlying `ArweaveAnchor.last_error` attribute carries the cause if you
  pass an explicit `arweave=` instance you can inspect.
- **Artifact hashing fails** (artifacts not yet logged, store unreachable):
  raises `ario_mlflow.anchoring.ArtifactAccessError`. Wrap the call if you
  want to log-and-continue.
- **Caller-supplied wallet missing or malformed** (when constructing your
  own `ArweaveAnchor(wallet_path=...)` and passing it as `arweave=`): raises
  `ario_mlflow.WalletLoadError` from the constructor — operator intent must
  not be silently overridden by an auto-generated wallet under a different
  on-chain identity. Pass `wallet_path=None` (or omit the arg) to use the
  auto-generated default.
- **No active run**: raises `RuntimeError`. The function requires an active
  `mlflow.start_run()` block.

### 2. `ario_mlflow.ArioMlflowClient` — registration + promotion

A drop-in replacement for `mlflow.tracking.MlflowClient`. Registration and stage
promotions are anchored automatically in a background thread. Query the outcome
via the client:

```python
from ario_mlflow import ArioMlflowClient

client = ArioMlflowClient()
mv = client.create_model_version("credit-scorer", "runs:/<run_id>/model")

# Block until the async anchor finishes (optional):
client.wait_for_anchor("registration", "credit-scorer", mv.version, timeout=30)

status = client.anchor_status("registration", "credit-scorer", mv.version)
# {"status": "anchored", "tx_id": "...", "error": None, "done": True}
```

**Failure modes.** Registration and promotion both return their MLflow
`ModelVersion` immediately; anchoring runs in a daemon thread.

- The MLflow operation always succeeds independently — anchoring failures
  never break `create_model_version()` or `transition_model_version_stage()`.
- `anchor_status()` returns `{"status": ...}` where status is one of
  `anchoring` (in flight), `anchored` (Arweave upload succeeded), `signed`
  (envelope signed but Arweave upload failed), `failed` (anchoring crashed —
  see `error`), or `unknown` (no anchor was ever queued for this key).
- `wait_for_anchor()` returns `False` on timeout. Process exit before the
  daemon completes is fine — the daemon is non-blocking by design.

### 3. `ario_mlflow.VerifiedModel` — inference

Wraps a registered model with an integrity check that runs **before** the
underlying pyfunc model is loaded (so a tampered artifact never gets a chance
to execute user code):

```python
from ario_mlflow import VerifiedModel

vm = VerifiedModel("models:/credit-scorer/1")  # raises IntegrityError on hash mismatch
# Features, in order: annual_income, credit_utilization, debt_to_income_ratio,
# months_employed, credit_score.
result = vm.predict([78000, 0.18, 0.22, 72, 745])
print(result.decision_id, result.proof_status)  # "anchoring" → "anchored"

# Wait for the background anchor if you want the TX synchronously:
result.wait_for_anchor(timeout=10)
print(result.tx_id, result.anchor_error)
```

**Failure modes.**

- **Tampered model artifact** — `VerifiedModel(model_uri)` raises
  `ario_mlflow.IntegrityError` *before* the underlying pyfunc model is loaded,
  so a swapped model never gets the chance to execute user code. Catch this
  exception to alert your security operations rather than silently fail open.
- **`predict()` always returns** the model's output even if anchoring later
  fails. Inspect `result.proof_status`: `anchoring` (in flight), `anchored`
  (Arweave upload succeeded), `failed` (see `result.anchor_error`), or
  `disabled` (no wallet / Turbo unavailable).
- **No registered model TX yet** — predictions chain to the model version's
  `ario.registration_tx`. If `ArioMlflowClient`'s registration daemon hasn't
  finished, the first few predictions chain to `GENESIS` (read once at model
  init; the registration TX never gets re-read on per-prediction calls — this
  avoids races).

#### Agent verify-status gate (runtime tamper detection)

When the model's *deployed* files are watched by the sister
[`ar-io-agent`](https://github.com/ar-io/ar-io-agent) daemon, pair
`VerifiedModel` with a `VerifyStatusClient` to consult the agent's verdict
before loading — `IntegrityError` covers the *registry* artifact; this gate
covers the *deployed* files the agent watches:

```python
from ario_mlflow import VerifiedModel, VerifyStatusClient

client = VerifyStatusClient(
    "http://127.0.0.1:9847",                                 # agent management port (loopback-only)
    secret=open("/var/lib/ario-agent/management-secret").read().strip(),
)

model = VerifiedModel(
    "models:/fraud-detector@production",
    asset_id="fraud-model",                                  # policy asset_id from the agent
    verify_status_client=client,
    on_failure="fail_closed",                                # "raise" | "fail_closed" | "fail_open"
    recheck_per_predict=True,                                # re-run the gate on every predict()
    recheck_max_cache_age=15.0,                              # contract §9.2 hot-path cache (10–30s)
)
model.predict(X)
```

The gate maps the §9.1 outcome vocabulary (`verified` / `tampered` /
`missing` / `unavailable` / `unknown` + transport-level errors) onto the
typed `AssetVerificationError` family — `IntegrityError` also subclasses it,
so one `except AssetVerificationError` clause catches both load-time gates.
`fail_open` logs a structured WARN with a `phase` field (`"load"` /
`"predict"`) for SIEM routing.

The endpoint is loopback-only by design; the `api_key=` constructor branch
is a forward-compatibility reservation with no server-side counterpart
today. Full failure-mode matrix in
[`docs/verified-model.md`](docs/verified-model.md).

## Dataset anchoring

Each MLflow dataset can have its own signed Arweave proof, independent of any
specific training run. Useful for:

- **Auditors** who need to prove "this dataset existed at time T, signed by X"
  without depending on a particular model run.
- **Dataset publishers** who anchor once and hand the TX to downstream model
  trainers.
- **Compliance** (e.g. EU AI Act Article 53 GPAI training-data summaries) that
  expects dataset-level artifacts, not fragments inside a model proof.

Two ways to use it:

```python
import mlflow
import ario_mlflow

ds = mlflow.data.from_pandas(df, source="s3://bucket/train_q1.parquet", name="train_q1")

# A) Implicit — auto-anchored inside training (recommended for typical use)
with mlflow.start_run():
    mlflow.log_input(ds, context="training")
    model.fit(...)
    mlflow.sklearn.log_model(model, "model")
    ario_mlflow.anchor()
    # Each logged dataset gets its own Arweave TX automatically;
    # the training proof references each by TX.

# B) Explicit — publisher pattern, no MLflow run needed
result = ario_mlflow.anchor(dataset=ds)
print(result["tx_id"])  # standalone dataset proof, hand off to downstream
```

The standalone-dataset envelope commits to the dataset's name, source URI,
digest, and schema hash — not to its rows. Datasets stay private; the
commitment is portable.

## Wallet & cost

Each anchored event is a ~500-byte signed commitment (bounded 400–700 bytes by
the plugin's smoke test). **Turbo's free tier covers uploads under 105 KiB**,
so typical usage is free — the auto-generated wallet works out of the box
with zero balance, and most teams never need to fund it.

The upload/funding wallet defaults to **Solana** (ed25519), persisted as a
Solana CLI `id.json` at `~/.ario-mlflow/wallet.json`. **Arweave RSA wallets
still work** — the chain is auto-detected from the key's shape, so a wallet
you already use keeps working unchanged. The anchor destination is Arweave
via Turbo either way; only the funding/signer chain differs.

You'd only need to fund the wallet if you're hitting Turbo's per-account
free-tier limits or anchoring larger payloads. To top up:

- Visit [console.ar.io](https://console.ar.io) — credit-card or crypto top-up
  for the wallet address logged by the plugin on first use
  (`wallet: <address>, mode=persistent`).

**For production deployments**, generate a dedicated wallet (don't rely on the
auto-generated one), point `ARIO_MLFLOW_ARWEAVE_WALLET` at it (a Solana
`id.json` / base58 secret **or** an Arweave RSA JWK — chain auto-detected),
and treat the wallet like any other production secret. Source data (params,
metrics, artifact bytes) always stays in MLflow — nothing else goes on chain —
so costs are flat regardless of how big your training run was.

**At scale.** Each event is one upload, so cost grows linearly with anchor
volume, not with model size. A high-throughput inference service anchoring
every prediction sees one ~500-byte upload per call — well under the per-file
free threshold. Account-level limits and any paid-tier rates are documented
at [console.ar.io](https://console.ar.io) and the
[ardrive.io Turbo docs](https://docs.ardrive.io); model your projected
volume against current rates before going live. See also
[`docs/plugin-production.md`](docs/plugin-production.md) for wallet
ops, monitoring, and balance alerting.

## Network requirements

If your environment restricts outbound traffic, allowlist:

| Host | Used for |
|---|---|
| `turbo-gateway.com` | Uploads (Turbo bundler) and proof fetches |
| `arweave.net` *or other ar.io gateways* | Proof fetches (fallback) |
| `turbo.ardrive.io` | TX bundler-status checks |
| Your configured `ARIO_MLFLOW_ARIO_VERIFY_URL` | Optional ar.io Verify attestations |

Override the upload/fetch host with `ARIO_MLFLOW_GATEWAY_HOST` if you want to
route through a specific gateway operator.

## Performance

What blocks vs what runs in the background:

| Call | Behavior |
|---|---|
| `anchor()` | **Synchronous.** Hashes artifacts, signs, uploads to Turbo before returning. Typically a few seconds end-to-end; longer if artifact hashing is large. |
| `ArioMlflowClient.create_model_version()` / `transition_model_version_stage()` | **Returns immediately**, anchors in a daemon thread. Use `wait_for_anchor()` if you need the TX before continuing. |
| `VerifiedModel.__init__` | **Synchronous.** Re-hashes artifacts, compares to `ario.artifact_hash`, raises `IntegrityError` on mismatch. One-time cost per model load. |
| `VerifiedModel.predict()` | **Returns immediately** with the prediction; anchor runs in a daemon thread. No per-prediction latency added by anchoring. |

For high-throughput inference, the predict path is the hot one — predictions
return as soon as the model produces an output. The Arweave upload happens
asynchronously and writes back to the trace tags when it completes.

## Resilience

The plugin's HTTP layer is built to absorb transient ar.io gateway
weather without bubbling up as user-visible failures.

- **Retries on transient failures.** All upload, fetch, and ar.io Verify
  requests share a `requests.Session` with a `urllib3` Retry adapter:
  HTTP 5xx and 429 responses are retried with exponential backoff
  (default: 2 retries, 0.5s/1.0s waits, `Retry-After` honored). 4xx
  responses other than 429 are not retried — they're hard failures.
  Tunable via `max_retries` and `retry_backoff_factor` constructor
  kwargs on `ArweaveAnchor` and `ArioVerifyClient`.
- **Multi-gateway fetch fallback.** `ArweaveAnchor.fetch_proof()` walks
  `self.gateways` in order: on a transient failure for one, the next is
  tried automatically. Default list is `["turbo-gateway.com",
  "ardrive.net"]`; override via the `gateways=` kwarg or the
  `ARIO_MLFLOW_GATEWAYS` env var. A single flaky gateway no longer
  surfaces as a hard "Proof Found" FAIL in any verifier UI.
- **Failure introspection via `last_error`.** When `upload_proof()`,
  `fetch_proof()`, or `ArioVerifyClient.submit_verification()` returns
  `None`, the instance's `last_error` attribute carries a string
  describing the cause — gateway down, retries exhausted, response
  body unparseable, etc. Distinguish "anchor disabled" from
  "everything we tried failed" without parsing logs.
- **Attestation-level polling.** `ArioVerifyClient.poll_attestation(tx_id,
  target_level=2, timeout=120, interval=5)` repeatedly submits the
  verification request until the desired attestation level is reached
  or the timeout expires. Returns the latest result either way (so
  callers can render "level 1, still propagating" status rather than
  nothing). Useful when you want to wait for full maturity before
  surfacing a Verified badge.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `ARIO_MLFLOW_ARWEAVE_WALLET` | Path to a funding-wallet key — a Solana `id.json` / base58 secret **or** an Arweave RSA JWK (chain auto-detected from the key shape) | auto-generates a **Solana** wallet + persists at `~/.ario-mlflow/wallet.json` |
| `ARIO_MLFLOW_GATEWAY_HOST` | Primary ar.io gateway used in returned URLs | `turbo-gateway.com` |
| `ARIO_MLFLOW_GATEWAYS` | Comma-separated list of ar.io gateways tried in order on fetch failures (e.g. `g1.com,g2.com`) | primary + `ardrive.net` fallback |
| `ARIO_MLFLOW_SIGNING_KEY` | Base64-encoded Ed25519 seed | auto-generates at `~/.ario-mlflow/keys/` |
| `ARIO_MLFLOW_ARIO_VERIFY_URL` | ar.io Verify REST API base URL — e.g. `https://perma.online/local/verify` (an ar.io operator's Verify endpoint) | ar.io attestation disabled if unset |

## Tags the plugin writes

On the training run (`anchor()`):

- `ario.enabled`, `ario.version` — via the registered `RunContextProvider`
- `ario.public_key`, `ario.verify_status`, `ario.artifact_hash`
- `ario.payload_hash` — SHA-256 of the canonical payload bytes (the same hash committed in the envelope)
- `ario.training_tx`, `ario.arweave_url` — when the Arweave upload succeeded
- `ario.wallet_mode` — `user-configured` / `persistent` / `ephemeral`

On the registered model (chain head, written by `anchor()`):

- `ario.last_training_hash` — pointer to the most recent training proof for this registered model; the next training reads it to set its `previous_hash`

On model versions (`ArioMlflowClient`):

- `ario.artifact_verified` — `true` / `false` from re-hashing at registration
- `ario.registration_tx`, `ario.promotion_tx`, `ario.arweave_url`

After running `ar-io-mlflow verify …` (training run or model version):

- `ario.verify_status` → `verified`
- `ario.attestation_level` — `1`, `2`, or `3` (see levels section below)
- `ario.report_url` — link to the ar.io Verify dashboard for this proof
- `ario.attested_by`, `ario.attested_at` — gateway operator and timestamp,
  only present when the operator has configured a signing wallet

On `@mlflow.trace` spans emitted by `VerifiedModel.predict()`:

- `ario.payload_json` — the full canonical payload (mirror of the
  `ario/predictions/<id>/payload.json` artifact). Read by `verify_source_of_truth`
  as the second MLflow surface for prediction check 3.
- `ario.decision_id`, `ario.model_name`, `ario.model_version`
- `ario.input_hash`, `ario.output_hash`, `ario.payload_hash`
- `ario.proof_status`, `ario.prediction_tx`, `ario.arweave_url`
- `ario.artifact_verified` (when known)

## CLI

```bash
ar-io-mlflow verify run <run_id>                  # verify training proof
ar-io-mlflow verify model <name>/<version>        # verify registration proof
ar-io-mlflow verify trace <trace_id>              # verify an inference proof
ar-io-mlflow audit <name>/<version>               # full model-lineage audit (terminal)
ar-io-mlflow audit <name>/<version> --format=json # machine-readable evidence bundle
ar-io-mlflow audit <name>/<version> --format=json --output lineage.json
```

`audit --format=json` emits an `ario.mlflow.audit/v1` evidence bundle — training → registration → promotion → artifact-integrity, each with per-check results and an `overall_ok` — for SOC2 / ISO 27001 evidence. It's the model-lineage parallel to the agent's `ariod audit export`. JSON mode is pipe-clean (no terminal panel); `--output` writes to a file.

The CLI reads `MLFLOW_TRACKING_URI` (default `./mlruns`) — export it to point
at the same store you used at training time, otherwise the run lookup will
fail with `Run '<id>' not found`. Set `ARIO_MLFLOW_ARIO_VERIFY_URL` to enable
the optional ar.io attestation row.

All `verify` commands run the same three-row verify flow plus the optional
ar.io attestation:

1. **Proof Found** — fetch the pure-commitment envelope from ar.io for the
   recorded TX ID.
2. **Decision / Training / Registration Record Matches** — download
   `ario/payload.json` from MLflow, re-hash, compare to the envelope's
   `payload_hash`, **and** re-derive canonical bytes from a *separate*
   live MLflow surface and compare to the anchored payload. This catches
   MLflow tampering — if either surface was modified after anchoring, the
   two won't agree.
   - `verify run` (`Training Record Matches`) re-fetches
     `run.data.params/metrics/artifact_checksums`.
   - `verify model` (`Registration Record Matches`) re-derives the
     artifact-verified state from the source run.
   - `verify trace` (`Decision Record Matches`) re-fetches the
     `ario.payload_json` trace tag (mirrored by `VerifiedModel.predict` at
     write time) and compares to the artifact.
3. **Signature Confirmed** — the signature on the envelope verifies
   against the embedded public key.

Plus an `Attested by` line — independent third-party check by an ar.io
gateway operator (when `ARIO_MLFLOW_ARIO_VERIFY_URL` is configured).

Results are written back to the MLflow tags and the HTML report is regenerated.

If an MLflow retention policy has pruned a prediction's trace, row 2 returns
`reason=live_refetch_incomplete` rather than silently passing — the proof
itself (signature + anchored bytes + ar.io) is on permanent storage and
remains verifiable.

## Programmatic verification

When you want to verify inside your own code — a CI gate, a scheduled
re-verification job, an auditor script — call the verify functions directly
instead of shelling out to the CLI. Three composite entry points, all
re-exported from the top-level `ario_mlflow` package; pick based on what you
hold and whether you have MLflow access:

| Function | Use when | MLflow access | Runs the live-MLflow check? |
|----------|----------|:---:|:---:|
| `verify_record(envelope, canonical_bytes, …)` | **Auditor** with a portable bundle (envelope + canonical bytes), no operator infra | no | no |
| `verify_proof_by_tx(tx_id, …)` | **Operator** with only a TX ID — fetches the envelope from Arweave, then verifies | optional | yes |
| `full_verify(envelope, …)` | **Operator** already holding the envelope | optional | yes |

```python
from mlflow.tracking import MlflowClient
from ario_mlflow import verify_proof_by_tx
from ario_mlflow.proof import ProofEngine        # not a top-level export
from ario_mlflow.arweave import ArweaveAnchor    # not a top-level export

result = verify_proof_by_tx(
    tx_id,                                # e.g. from the ario.training_tx tag
    anchor=ArweaveAnchor("", "turbo-gateway.com"),
    proof_engine=ProofEngine(),
    mlflow_client=MlflowClient(),         # enables the anchored-bytes + live checks
)
assert result["proof_found"] and result["overall"] is True
```

**Read results carefully — `ok` is tri-state.** Each check reports
`ok=True` (passed), `ok=False` (ran and failed), or `ok=None` (didn't run /
not applicable, with a `reason`). One sharp edge: for training, registration,
and prediction events the MLflow checks are *required*, so calling
`full_verify` / `verify_proof_by_tx` **without** an `mlflow_client` returns
`overall=False` even on a valid signature — a missing check can't read as a
pass. For a deliberate offline/signature-only verdict, use `verify_record`
(auditor semantics) or inspect `result["signature"]["ok"]`.

The plugin also verifies envelopes minted by the sister
[`ar-io-agent`](https://github.com/ar-io/ar-io-agent) daemon
(`ario.agent/v1`) — the two share the envelope spec and crypto.

See [`docs/verification.md`](docs/verification.md) for the full reference:
every result field, per-event-type nuances, the `ario/verification.html`
report, `spec_version` handling, and a copy-paste CI/monitoring job.

## What the ar.io attestation means

`ar-io-mlflow verify` reports the ar.io attestation as `Verified` or
`Pending verification`. A proof reads `Verified` once an ar.io gateway has:

1. Found it permanently stored on Arweave.
2. Re-downloaded the bytes and matched the SHA-256 against the gateway's own digest.
3. Verified the signature against the original signer's public key.

For programmatic callers, `ario.attestation_level` exposes the same status as
an integer (1, 2, or 3) — useful when you want to distinguish "still
propagating" from "fully verified."

**Operator attestation.** When the ar.io gateway operator has configured a
signing wallet, the verification result is itself signed and
`ario.attested_by` / `ario.attested_at` get written back to your MLflow tags.
That's an independent statement from a known operator, verifiable by any
third party against their public key (standard RSA-PSS SHA-256).

These attestations cover **integrity and authenticity** of the anchored
record. Semantic verification (whether *this model* produced *this output*
on *this input*) is a separate problem the plugin intentionally doesn't
solve — out of scope by design, not a pending feature.

## Verifying without Python

The proof envelope spec is language-neutral: an Ed25519 signature over an
RFC-8785 (JCS) canonicalized JSON object, with a SHA-256 commitment to the
canonical payload bytes that live as an MLflow artifact. Any RFC-8785 +
Ed25519 + SHA-256 implementation in any language can verify a proof — no
`ar-io-mlflow` install needed.

The auditor recipe:

1. **Fetch the envelope** from any ar.io gateway: `GET https://<gateway>/raw/<tx_id>`.
2. **Verify the signature.** Strip the `signature` field from the envelope,
   JCS-canonicalize the rest (RFC 8785), then verify the original
   `signature` (hex) against the embedded `public_key` (hex) using Ed25519.
3. **Re-hash the canonical payload.** Download `ario/payload.json` from the
   MLflow run's artifacts. Compute SHA-256 of the raw bytes. Compare to the
   envelope's `payload_hash`.
4. **Check `spec_version`.** Accept the `ario.mlflow/v*` / `ario.agent/v*`
   majors you understand and reject unknown ones. Envelopes anchored before
   this field existed have no `spec_version` and still verify (legacy) — see
   [`docs/architecture.md`](docs/architecture.md#pure-commitment-proofs-500-bytes-on-arweave).
5. **Walk the chain** (optional). Each envelope's `previous_hash` is the
   prior anchor's `payload_hash` for that event type, or `"GENESIS"`.
   Fetch the predecessor by its TX (recorded in the relevant tag,
   e.g. `ario.last_training_hash`) and recurse.

JCS implementations exist for Python (`jcs`), JavaScript (`canonicalize`),
Go (`gowebpki/jcs`), Java, Rust, and others — interoperable with the same
ecosystem as Notary and Sigstore.

The Python plugin is a convenience wrapper around this recipe; the proof
itself doesn't depend on the plugin's continued existence.

## Tests

```bash
pytest tests/test_plugin_smoke.py tests/test_plugin_verify.py tests/test_input_anchoring.py
```

No network or MLflow server required.

## Related docs

- [`CHANGELOG.md`](CHANGELOG.md) — release history and known limitations.
- [`docs/verification.md`](docs/verification.md) — full verification reference: the four checks, the programmatic API, result shapes, the HTML report, and a CI/monitoring recipe.
- [`docs/architecture.md`](docs/architecture.md) — system design (pure-commitment proofs, per-event chains, JCS canonicalization).
- [`docs/plugin-production.md`](docs/plugin-production.md) — production deployment guide: wallet ops, CI/CD patterns, monitoring, runbooks.
- [`docs/plugin-threat-model.md`](docs/plugin-threat-model.md) — what the plugin defends against, what it doesn't, trust boundaries.
- A reference demo app using this plugin lives at
  [vilenarios/Verifiable-AI-Decision-Records-Demo](https://github.com/vilenarios/Verifiable-AI-Decision-Records-Demo).
