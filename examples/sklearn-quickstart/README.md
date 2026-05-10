# sklearn quickstart

Smallest possible example of `ario-mlflow`: train a toy sklearn model, log it
to a local MLflow store, and anchor a pure-commitment proof to Arweave.

## Run

From the repo root:

```bash
pip install -e .                    # installs the plugin
pip install scikit-learn            # if not already installed
python examples/sklearn-quickstart/train_and_anchor.py
```

The script creates `examples/sklearn-quickstart/mlruns-quickstart/` alongside
itself, trains a Logistic Regression on iris, logs the model + metric, and
calls `ario_mlflow.anchor()`.

## What you should see

- One log line from the plugin telling you where the auto-generated Arweave
  wallet was persisted (first run only).
- The run ID, accuracy, wallet mode, verify status, artifact hash, the new
  `payload_hash` (commitment hash), and (if Arweave anchoring succeeded) the
  Arweave TX.
- A size comparison showing the on-Arweave envelope (~500 bytes, signed
  commitment) vs. the canonical payload (in MLflow as `ario/payload.json`).
- Two copy-paste commands — one to verify the four checks with the CLI, one
  to open the MLflow UI to see the `ario.*` tags + `ario/` artifacts.

## What's happening under the hood

1. `mlflow.sklearn.log_model(model, "model")` writes the pickled model and a
   conda env to the run's artifacts.
2. `ario_mlflow.anchor()`:
   - Builds a canonical payload from the run's params/metrics/artifact
     checksums plus any caller `metadata`.
   - Writes the canonical bytes as `ario/payload.json` MLflow artifact —
     the witness a verifier downloads to recompute the hash.
   - Signs a small commitment envelope (event_id, subject, payload_hash,
     previous_hash, signed_at, public_key, signature) and uploads it
     to Arweave via ar.io Turbo.
   - Sets `ario.training_tx`, `ario.payload_hash`, `ario.artifact_hash`
     on the run; if a registered model points to this run, sets
     `ario.last_training_hash` on it so the next training chains here.
3. `ario-mlflow verify run <run_id>` runs the four-check verification:
   - **Signature** valid? (Ed25519 over the envelope)
   - **Anchored bytes intact?** (download `ario/payload.json`, hash,
     compare to `payload_hash`)
   - **Live MLflow matches anchored bytes?** (re-fetch params/metrics/
     checksums from MLflow, rebuild the canonical payload, compare bytes
     — catches MLflow-side tampering)
   - **(Optional) ar.io Verify Level 3 attestation** — if
     `ARIO_MLFLOW_ARIO_VERIFY_URL` is set.

## Correlating with OpenTelemetry (zero config)

OpenTelemetry trace IDs are auto-captured by default whenever a
recording span is active. No code changes required — wrap your call in
your existing tracer and the OTel IDs flow into the signed commitment:

```python
from opentelemetry import trace
import ario_mlflow

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("train_fraud_model"):
    with mlflow.start_run():
        # ... fit, log_model ...
        ario_mlflow.anchor()  # otel_trace_id + otel_span_id auto-captured
```

The OTel IDs become part of the canonical bytes that get hashed and
signed, so verification is end-to-end: an SRE looking at a flagged
trace in Datadog/Jaeger can pull the proof for that `otel_trace_id`,
hash the canonical payload, and confirm it matches what was anchored.

`VerifiedModel.predict()` and `ArioMlflowClient.create_model_version()`
auto-capture the same way — wrap them in your existing OTel spans and
the correlation flows through automatically.

**Opting out:**

- Per-call: `ario_mlflow.anchor(capture_otel=False)`,
  `model.predict(features, capture_otel=False)`, etc.
- Process-wide: set the env var `ARIO_MLFLOW_CAPTURE_OTEL=false`.

Either disables auto-capture; OTel IDs are then only included if you
pass them explicitly via `metadata={"otel_trace_id": ...,
"otel_span_id": ...}`. Caller-supplied values always win over auto-
captured ones, so you can also override on a per-call basis without
disabling.

**Soft dependency:** `opentelemetry-api` is not in `install_requires`
— if it's not installed, auto-capture silently no-ops. Most production
ML services already have it.

## Reset between runs

Delete the local MLflow store to start fresh:

```bash
rm -rf examples/sklearn-quickstart/mlruns-quickstart
```
