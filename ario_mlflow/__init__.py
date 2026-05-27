"""ar.io MLflow plugin — verifiable provenance for the ML lifecycle.

Public API:

- :func:`anchor` — call inside ``mlflow.start_run()`` to anchor a
  training-complete commitment to Arweave.
- :class:`VerifiedModel` — wraps an MLflow model with load-time
  integrity check + per-prediction commitments.
- :class:`ArioMlflowClient` — drop-in ``MlflowClient`` subclass that
  auto-anchors registration and promotion events.
- :class:`IntegrityError` — raised by ``VerifiedModel`` when artifact
  integrity fails.
- :class:`WalletLoadError` — raised when a caller-supplied Arweave
  wallet path cannot be loaded. The plugin refuses to silently sign
  with an auto-generated wallet under a different identity.
- :func:`verify_signature`, :func:`verify_anchored_bytes`,
  :func:`verify_source_of_truth`, :func:`verify_ario_attestation`,
  :func:`full_verify` — the four-check verification helpers. Re-exported
  here so consumers can import them without spelunking
  ``ario_mlflow.verify``.
- :func:`verify_record` — auditor-shaped foundation primitive. Given an
  envelope plus its canonical bytes, runs signature + payload-hash
  match + optional ar.io attestation. No MLflow access required —
  what an auditor uses against a portable bundle.
- :func:`verify_proof_by_tx` — operator-side wrapper. Fetches the
  envelope from Arweave by TX, then runs all four checks (composing
  ``verify_record`` with the live MLflow source-of-truth check). Adds
  ``proof_found`` so callers can distinguish "envelope retrieved" from
  "envelope was missing."
- :class:`ArioVerifyClient` — ar.io Verify REST client.
"""

__version__ = "0.2.1"


def __getattr__(name):
    if name == "SPEC_VERSION":
        from ario_mlflow.proof import SPEC_VERSION
        return SPEC_VERSION
    if name == "anchor":
        from ario_mlflow.anchoring import anchor
        return anchor
    if name == "VerifiedModel":
        from ario_mlflow.model import VerifiedModel
        return VerifiedModel
    if name == "IntegrityError":
        from ario_mlflow.model import IntegrityError
        return IntegrityError
    if name == "WalletLoadError":
        from ario_mlflow.arweave import WalletLoadError
        return WalletLoadError
    if name == "ArioMlflowClient":
        from ario_mlflow.client import ArioMlflowClient
        return ArioMlflowClient
    if name in (
        "verify_signature",
        "verify_anchored_bytes",
        "verify_source_of_truth",
        "verify_ario_attestation",
        "full_verify",
        "verify_record",
        "verify_proof_by_tx",
        "ArioVerifyClient",
    ):
        from ario_mlflow import verify as _verify
        return getattr(_verify, name)
    raise AttributeError(f"module 'ario_mlflow' has no attribute {name!r}")


__all__ = [
    "__version__",
    "SPEC_VERSION",
    "anchor",
    "VerifiedModel",
    "IntegrityError",
    "WalletLoadError",
    "ArioMlflowClient",
    "verify_signature",
    "verify_anchored_bytes",
    "verify_source_of_truth",
    "verify_ario_attestation",
    "full_verify",
    "verify_record",
    "verify_proof_by_tx",
    "ArioVerifyClient",
]
