"""Minimal end-to-end example: train a model, anchor a commitment, verify it.

Run:
    python examples/sklearn-quickstart/train_and_anchor.py

The script trains a toy classifier, logs it to a local MLflow store at
./mlruns-quickstart, calls ``ario_mlflow.anchor()`` to create a signed
pure-commitment proof, and prints the CLI command you can run to
verify the four checks (signature / anchored bytes / live MLflow /
ar.io Verify) after the fact.

What anchor() does (per the redesign):
1. Builds a canonical payload from the run's params, metrics, and
   artifact checksums (plus any caller-supplied ``metadata=...``).
2. Writes the canonical bytes as ``ario/payload.json`` artifact on the
   MLflow run — the witness a verifier downloads to recompute the
   hash.
3. Signs a tiny ~500-byte envelope (only event_id, event_type,
   subject, payload_hash, previous_hash, signed_at, public_key,
   signature). Uploads to Arweave.
4. Updates ``ario.last_training_hash`` on the registered model (if
   any) so the next training of this model chains to this proof.

No Arweave wallet is required — the plugin will auto-generate one at
``~/.ario-mlflow/wallet.json`` and reuse it on subsequent runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import mlflow.data
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import ario_mlflow
from ario_mlflow.proof import canonical_json

TRACKING_URI = Path(__file__).parent / "mlruns-quickstart"


def main() -> int:
    mlflow.set_tracking_uri(f"file://{TRACKING_URI.resolve()}")
    mlflow.set_experiment("ar-io-mlflow-quickstart")

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        # Log the training dataset so anchor() can mint its own
        # standalone dataset proof referenced from the training event.
        train_ds = mlflow.data.from_numpy(
            X_train,
            targets=y_train,
            source="https://archive.ics.uci.edu/dataset/53/iris",
            name="iris-train",
        )
        mlflow.log_input(train_ds, context="training")

        model = LogisticRegression(max_iter=200).fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        mlflow.log_params({"max_iter": 200, "random_state": 42})
        mlflow.log_metric("accuracy", round(acc, 6))
        mlflow.sklearn.log_model(model, "model")

        # Caller metadata flows into the canonical payload alongside
        # the MLflow data. Useful for service identity, custom
        # compliance fields, etc. Structural fields and the auto-
        # captured OTel context can't be overwritten silently —
        # caller-supplied keys win on collision, structural fields
        # always win.
        #
        # OTel context (otel_trace_id, otel_span_id) is captured
        # automatically when a recording span is active — no need to
        # pass it through metadata. See README "Correlating with
        # OpenTelemetry" for details + opt-out.
        result = ario_mlflow.anchor(
            metadata={"service_name": "ar-io-mlflow-quickstart"},
        )

    print()
    print("Run ID:              ", run.info.run_id)
    print("Accuracy:            ", round(acc, 4))
    print("Wallet mode:         ", result["tags"].get("ario.wallet_mode", "unknown"))
    print("Verify status:       ", result["tags"]["ario.verify_status"])
    print("Artifact status:     ", result["artifact_status"])
    print("Artifact hash:       ", result["tags"].get("ario.artifact_hash", "n/a"))
    print("Payload hash:        ", result["payload_hash"])
    print("Chain previous:      ", result["previous_hash"])
    if "ario.training_tx" in result["tags"]:
        print("Arweave TX:          ", result["tags"]["ario.training_tx"])
        print("Arweave URL:         ", result["tags"]["ario.arweave_url"])

    # The on-Arweave envelope is much smaller than the canonical
    # payload — that's the privacy-preserving point of the redesign.
    envelope_size = len(canonical_json(result["envelope"]))
    payload_size = len(result["payload_bytes"])
    print()
    print(f"Envelope size:        {envelope_size} bytes (signed, on Arweave)")
    print(f"Canonical payload:    {payload_size} bytes (in MLflow as ario/payload.json)")

    print()
    print("Verify the four checks later with:")
    print(f"  MLFLOW_TRACKING_URI={TRACKING_URI.resolve()} \\")
    print(f"  ar-io-mlflow verify run {run.info.run_id}")
    print()
    print("Open the MLflow UI to see the ario.* tags + ario/ artifacts:")
    print(f"  mlflow ui --backend-store-uri {TRACKING_URI.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
