"""CLI: ar-io-mlflow verify and audit commands.

Updated for the pure-commitment redesign \u2014 runs the three-row verify
flow (Proof Found / {Event} Record Matches / Signature Confirmed) plus an
optional ar.io attestation row, matching the dashboard vocabulary.

Internal field names in ``ario_mlflow.verify`` (``signature_valid``,
``hash_match``, ``source_of_truth_ok``, ``attestation_level``,
``permanent_copy_found``) are stable API and unchanged \u2014 only the
printed labels match the UI.
"""

import argparse
import json
import os
import sys
import tempfile

import mlflow

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.arweave import ArweaveAnchor
from ario_mlflow.verify import (
    ArioVerifyClient,
    verify_signature,
    verify_anchored_bytes,
    verify_source_of_truth,
    verify_ario_attestation,
    _compute_overall_ok,
)
from ario_mlflow.report import generate_verification_html


def _get_components():
    proof_engine = ProofEngine()
    anchor = ArweaveAnchor(
        os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", ""),
        os.environ.get("ARIO_MLFLOW_GATEWAY_HOST", "turbo-gateway.com"),
    )
    ario = ArioVerifyClient()
    return proof_engine, anchor, ario


_LABEL_WIDTH = 28

def _glyph(symbol: str, color_code: str) -> str:
    """Return ``symbol`` wrapped in an ANSI color code.

    Respects the NO_COLOR community standard (https://no-color.org): when
    the ``NO_COLOR`` env var is set to any non-empty value, ANSI escapes
    are stripped \u2014 useful for piping ``ar-io-mlflow verify`` into log
    files, CI artifacts, or downstream parsers that mangle escape codes.
    Read at call time (not import time) so tests can monkeypatch the
    env var without reloading the module.
    """
    if os.environ.get("NO_COLOR", ""):
        return symbol
    return f"\033[{color_code}m{symbol}\033[0m"


def _check_glyph() -> str:
    return _glyph("\u2713", "32")


def _cross_glyph() -> str:
    return _glyph("\u2717", "31")


def _pending_glyph() -> str:
    return _glyph("?", "33")


def _print_check(label: str, result: dict, value_when_ok: str | None = None):
    """Print one check line. Status: \u2713 / \u2717 / ? (not applicable)."""
    check = _check_glyph()
    cross = _cross_glyph()
    pending = _pending_glyph()

    ok = result.get("ok")
    if ok is True:
        symbol = check
        suffix = value_when_ok or "passed"
    elif ok is False:
        symbol = cross
        suffix = result.get("reason", "FAILED")
    else:
        symbol = pending
        suffix = result.get("reason", "not applicable")
    print(f"  {label:<{_LABEL_WIDTH}} {symbol} {suffix}")


def _print_four_checks(
    sig: dict,
    bytes_check: dict,
    sot: dict,
    ario_attestation: dict,
    record_label: str = "Record Matches",
):
    """Print the verify panel for a single envelope.

    Three rows match the dashboard vocabulary:
      1. Proof Found            (envelope retrieved from ar.io)
      2. {record_label}         ("Decision Record Matches" /
                                 "Training Record Matches" /
                                 "Registration Record Matches" \u2014 passed
                                 in by the caller depending on the
                                 event type). Consolidates the legacy
                                 "anchored bytes intact" + "live MLflow
                                 source-of-truth" checks into one row.
      3. Signature Confirmed    (signature on envelope verifies)

    Plus the ar.io attestation line \u2014 always surfaces the maturity
    level when present (even on a "below threshold" failure), so the
    user can see how the TX is maturing. Per ROADMAP "Receipts vs.
    attestation as a two-stage verify UX", attestation level is
    fundamentally a maturity gradient, not a binary pass/fail.
    """
    # Row 1: Proof Found \u2014 by the time we got here we already fetched
    # the envelope, so this is a synthetic ok=True row that keeps the
    # CLI structurally aligned with the dashboard's three-row card.
    _print_check("Proof Found", {"ok": True}, "envelope retrieved from ar.io")

    # Row 2: consolidated "{record_label}" \u2014 the new vocabulary folds
    # the old `Anchored bytes` + `Source of truth` into one row, but
    # internally we still verify both. Surface a fail if either side
    # failed, surface pending if either is None.
    consolidated = _consolidate_record_check(bytes_check, sot)
    _print_check(record_label, consolidated, "live MLflow matches anchored bytes")

    # Row 3: Signature Confirmed
    _print_check("Signature Confirmed", sig, "signature valid")

    # ar.io Verify \u2014 show the maturity level whenever the API returned
    # something, regardless of whether it passed the threshold. Helps
    # users see "Level 1, growing" vs. "TX missing" at a glance.
    check = _check_glyph()
    cross = _cross_glyph()
    pending = _pending_glyph()
    ok = ario_attestation.get("ok")
    attester = ario_attestation.get("attested_by") or "unknown"

    if ok is True:
        # User-facing copy collapses to "Verified" \u2014 internal
        # attestation_level is preserved for programmatic callers.
        print(f"  {'ar.io attestation':<{_LABEL_WIDTH}} {check} Verified by {attester}")
        if ario_attestation.get("report_url"):
            print(f"  {'':>{_LABEL_WIDTH}} report: {ario_attestation['report_url']}")
    elif ok is False and ario_attestation.get("reason") == "attestation_level_below_threshold":
        # TX indexed but maturity not yet at the configured bar. This is
        # transient propagation, not a verification failure — render with
        # the yellow "pending" glyph rather than a red cross.
        print(
            f"  {'ar.io attestation':<{_LABEL_WIDTH}} {pending} Pending verification "
            f"(by {attester})"
        )
        print(
            f"  {'':>{_LABEL_WIDTH}} TX is indexed but not yet at the configured "
            f"attestation bar. Re-run later to check progression."
        )
        if ario_attestation.get("report_url"):
            print(f"  {'':>{_LABEL_WIDTH}} report: {ario_attestation['report_url']}")
    elif ok is False:
        print(
            f"  {'ar.io attestation':<{_LABEL_WIDTH}} {cross} "
            f"{ario_attestation.get('reason', 'FAILED')}"
        )
    else:
        # ok is None: not applicable / not checked
        print(
            f"  {'ar.io attestation':<{_LABEL_WIDTH}} {pending} "
            f"{ario_attestation.get('reason', 'not checked')}"
        )


def _consolidate_record_check(bytes_check: dict, sot: dict) -> dict:
    """Combine the anchored-bytes and source-of-truth checks into one row.

    Truth table:
      both ok=True  \u2192 ok=True
      either False  \u2192 ok=False (with the failing reason)
      either None   \u2192 ok=None (with the unknown reason)
    Internal field names (``signature_valid``, ``hash_match``,
    ``source_of_truth_ok``) in the underlying dict stay untouched \u2014
    this only affects the printed row.
    """
    bytes_ok = bytes_check.get("ok")
    sot_ok = sot.get("ok")

    if bytes_ok is False:
        return {"ok": False, "reason": bytes_check.get("reason", "anchored_bytes_mismatch")}
    if sot_ok is False:
        return {"ok": False, "reason": sot.get("reason", "live_mlflow_mismatch")}
    if bytes_ok is None:
        return {"ok": None, "reason": bytes_check.get("reason", "no_payload_to_compare")}
    if sot_ok is None:
        return {"ok": None, "reason": sot.get("reason", "live_refetch_incomplete")}
    return {"ok": True}


def _verify_envelope_for_tx(
    tx_id: str,
    proof_engine: ProofEngine,
    anchor: ArweaveAnchor,
    ario_client: ArioVerifyClient,
    mlflow_client,
    record_label: str = "Record Matches",
) -> tuple[dict, bool]:
    """Fetch an envelope from ar.io and run the verify checks.

    Returns ``(combined_result, overall_ok)``. ``combined_result`` has
    keys ``signature`` / ``anchored_bytes`` / ``source_of_truth`` /
    ``ario_attestation`` for callers that want to programmatically
    inspect; the printed output goes to stdout.

    ``record_label`` controls the row-2 label in the printed panel
    ("Decision Record Matches" / "Training Record Matches" /
    "Registration Record Matches"). Internal field names are unchanged.
    """
    envelope = anchor.fetch_proof(tx_id)
    if not envelope:
        print(f"  Could not fetch envelope from ar.io for TX {tx_id}.")
        return {}, False

    sig = verify_signature(envelope, proof_engine)
    bytes_check = verify_anchored_bytes(envelope, mlflow_client)
    sot = (
        verify_source_of_truth(envelope, bytes_check.get("payload_bytes") or b"", mlflow_client)
        if bytes_check.get("payload_bytes")
        else {"ok": None, "reason": bytes_check.get("reason", "no_payload_to_compare")}
    )
    # For ar.io Verify, inject the TX ID the caller already knows. The
    # envelope itself doesn't carry it (the TX IS its address).
    envelope_with_tx = dict(envelope)
    envelope_with_tx["_tx_id"] = tx_id
    ario_result = verify_ario_attestation(envelope_with_tx, ario_client)

    _print_four_checks(sig, bytes_check, sot, ario_result, record_label=record_label)

    # Use the shared overall-ok logic so CLI and full_verify() agree.
    # For training/registration envelopes, ok=None on signature /
    # anchored_bytes / source_of_truth fails overall \u2014 None means
    # "couldn't verify", not "fine."
    overall = _compute_overall_ok(envelope, sig, bytes_check, sot, ario_result)
    overall_ok = bool(overall)  # CLI returns bool; None coerces to False

    return {
        "envelope": envelope,
        "signature": sig,
        "anchored_bytes": bytes_check,
        "source_of_truth": sot,
        "ario_attestation": ario_result,
    }, overall_ok


def _verification_run_tags(verification: dict | None) -> dict[str, str]:
    """Map a normalized ar.io Verify result to MLflow tag key/values."""
    tags: dict[str, str] = {}
    if not verification:
        return tags
    level = verification.get("attestation_level")
    if level is not None:
        tags["ario.verify_status"] = "verified"
        tags["ario.attestation_level"] = str(level)
    if verification.get("report_url"):
        tags["ario.report_url"] = verification["report_url"]
    if verification.get("attested_by"):
        tags["ario.attested_by"] = verification["attested_by"]
    if verification.get("attested_at"):
        tags["ario.attested_at"] = verification["attested_at"]
    return tags


def _regenerate_html(
    run_id: str,
    proof: dict,
    tx_id: str,
    arweave_url: str | None,
    artifact_hash: str | None,
    artifact_verified: bool | None,
    verification: dict | None,
    filename: str,
    cli_verify_cmd: str | None = None,
):
    """Regenerate an ario/<filename> artifact on a run with updated verification."""
    anchor_result = {"tx_id": tx_id, "url": arweave_url or "", "receipt": None}
    html_content = generate_verification_html(
        proof,
        anchor_result,
        artifact_hash=artifact_hash,
        artifact_verified=artifact_verified,
        verification=verification,
        cli_verify_cmd=cli_verify_cmd,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        ario_dir = os.path.join(tmpdir, "ario")
        os.makedirs(ario_dir)
        with open(os.path.join(ario_dir, filename), "w") as f:
            f.write(html_content)
        client = mlflow.tracking.MlflowClient()
        client.log_artifacts(run_id, ario_dir, "ario")


def cmd_verify_run(args):
    """Verify a training run's commitment (Proof Found / Training Record Matches / Signature Confirmed)."""
    proof_engine, anchor, ario_client = _get_components()
    client = mlflow.tracking.MlflowClient()

    run = client.get_run(args.run_id)
    tx_id = run.data.tags.get("ario.training_tx")

    if not tx_id:
        print(f"Run {args.run_id}: no ario.training_tx tag found. Not anchored.")
        return 1

    print(f"Verifying training run {args.run_id}")
    print(f"  TX: {tx_id}")

    result, ok = _verify_envelope_for_tx(
        tx_id, proof_engine, anchor, ario_client, client,
        record_label="Training Record Matches",
    )
    if not result:
        return 1

    ario = result.get("ario_attestation", {})
    tags = _verification_run_tags(ario)
    if tags:
        for key, value in tags.items():
            client.set_tag(args.run_id, key, value)
        print(f"  -> updated {len(tags)} MLflow tag(s) on run")

    return 0 if ok else 1


def cmd_verify_model(args):
    """Verify a model version's registration commitment (Proof Found / Registration Record Matches / Signature Confirmed)."""
    proof_engine, anchor, ario_client = _get_components()
    client = mlflow.tracking.MlflowClient()

    parts = args.model.split("/")
    name = parts[0]
    version = parts[1] if len(parts) > 1 else "1"

    mv = client.get_model_version(name, version)
    tx_id = mv.tags.get("ario.registration_tx")

    if not tx_id:
        print(f"Model {name}/v{version}: no ario.registration_tx tag found. Not anchored.")
        return 1

    print(f"Verifying model registration {name}/v{version}")
    print(f"  TX: {tx_id}")

    result, ok = _verify_envelope_for_tx(
        tx_id, proof_engine, anchor, ario_client, client,
        record_label="Registration Record Matches",
    )
    if not result:
        return 1

    ario = result.get("ario_attestation", {})
    tags = _verification_run_tags(ario)
    if tags:
        for key, value in tags.items():
            client.set_model_version_tag(name, version, key, value)
        print(f"  -> updated {len(tags)} MLflow tag(s) on model version")

    return 0 if ok else 1


def cmd_verify_trace(args):
    """Verify a prediction trace's commitment (Proof Found / Decision Record Matches / Signature Confirmed).

    Note: predictions don't write a payload.json artifact (per the
    privacy-preserving design — canonical fields are mirrored as trace
    tags). When that's the case, the consolidated "Decision Record
    Matches" row reports as not-applicable. Proof Found, Signature
    Confirmed, and the ar.io attestation row work normally. Auditors
    with the raw input/output can verify input_hash / output_hash
    directly.
    """
    proof_engine, anchor, ario_client = _get_components()
    client = mlflow.tracking.MlflowClient()

    try:
        trace = mlflow.get_trace(args.trace_id)
    except Exception as e:  # noqa: BLE001 — CLI command: any trace-load failure (missing, network, parse) returns exit 1 with a user-readable error
        print(f"Could not load trace {args.trace_id}: {e}")
        return 1

    if trace is None:
        print(f"Trace {args.trace_id} not found.")
        return 1

    tags = getattr(trace.info, "tags", {}) or {}
    # Prefer the new tag name (ario.prediction_tx); accept the legacy
    # (ario.arweave_tx) for backwards compat with any traces anchored
    # under v1 still in MLflow.
    tx_id = tags.get("ario.prediction_tx") or tags.get("ario.arweave_tx")

    if not tx_id:
        print(f"Trace {args.trace_id}: no ario.prediction_tx tag found. Not anchored yet.")
        return 1

    print(f"Verifying trace {args.trace_id}")
    print(f"  TX: {tx_id}")

    result, ok = _verify_envelope_for_tx(
        tx_id, proof_engine, anchor, ario_client, client,
        record_label="Decision Record Matches",
    )
    if not result:
        return 1

    ario = result.get("ario_attestation", {})
    back_tags = _verification_run_tags(ario)
    if back_tags:
        for key, value in back_tags.items():
            try:
                mlflow.set_trace_tag(args.trace_id, key, value)
            except Exception as e:  # noqa: BLE001 — best-effort tag write-back; CLI continues with verification result even if a single tag setter fails
                print(f"  ! failed to set trace tag {key}: {e}")
        print(f"  -> updated {len(back_tags)} MLflow trace tag(s)")

    return 0 if ok else 1


def cmd_audit(args):
    """Audit the full lineage (training → registration → promotion) for a model version."""
    proof_engine, anchor, ario_client = _get_components()
    client = mlflow.tracking.MlflowClient()

    parts = args.model.split("/")
    name = parts[0]
    version = parts[1] if len(parts) > 1 else "1"

    print(f"Auditing model lineage: {name}/v{version}")
    print("=" * 50)

    mv = client.get_model_version(name, version)
    all_ok = True

    # 1. Training
    training_tx = None
    if mv.run_id:
        try:
            run = client.get_run(mv.run_id)
            training_tx = run.data.tags.get("ario.training_tx")
        except Exception:  # noqa: BLE001 — audit display: training_tx is best-effort; missing tag/run renders as "unknown" without aborting the audit
            pass

    print(f"\nTraining (run {mv.run_id or 'unknown'}):")
    if training_tx:
        _, ok = _verify_envelope_for_tx(
            training_tx, proof_engine, anchor, ario_client, client,
            record_label="Training Record Matches",
        )
        if not ok:
            all_ok = False
    else:
        print("  Not anchored.")

    # 2. Registration
    registration_tx = mv.tags.get("ario.registration_tx")
    print(f"\nRegistration (v{version}):")
    if registration_tx:
        _, ok = _verify_envelope_for_tx(
            registration_tx, proof_engine, anchor, ario_client, client,
            record_label="Registration Record Matches",
        )
        if not ok:
            all_ok = False
    else:
        print("  Not anchored.")

    # 3. Promotion
    promotion_tx = mv.tags.get("ario.promotion_tx")
    print(f"\nPromotion ({mv.current_stage}):")
    if promotion_tx:
        _, ok = _verify_envelope_for_tx(
            promotion_tx, proof_engine, anchor, ario_client, client,
            record_label="Registration Record Matches",
        )
        if not ok:
            all_ok = False
    else:
        print("  Not anchored.")

    # 4. Artifact integrity
    artifact_hash = None
    if mv.run_id:
        try:
            run = client.get_run(mv.run_id)
            artifact_hash = run.data.tags.get("ario.artifact_hash")
        except Exception:  # noqa: BLE001 — audit display: artifact_hash is best-effort; failure renders as "unknown" without aborting
            pass

    print(f"\nArtifact integrity:")
    if artifact_hash:
        print(f"  Anchored hash: {artifact_hash[:24]}...")
    else:
        print("  No artifact hash recorded.")

    print(f"\n{'=' * 50}")
    check = _check_glyph()
    cross = _cross_glyph()
    print(f"Overall: {check + ' All checks passed' if all_ok else cross + ' Issues found'}")
    return 0 if all_ok else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser. Exposed so tests can exercise the real wiring."""
    parser = argparse.ArgumentParser(prog="ar-io-mlflow", description="ar.io MLflow verification CLI")
    subparsers = parser.add_subparsers(dest="command")

    # verify
    verify_parser = subparsers.add_parser("verify", help="Verify a proof record")
    verify_sub = verify_parser.add_subparsers(dest="verify_type")

    run_parser = verify_sub.add_parser("run", help="Verify a training run")
    run_parser.add_argument("run_id", help="MLflow run ID")

    model_parser = verify_sub.add_parser("model", help="Verify a model registration")
    model_parser.add_argument("model", help="Model name/version (e.g. fraud-detector/3)")

    trace_parser = verify_sub.add_parser("trace", help="Verify an inference trace")
    trace_parser.add_argument("trace_id", help="MLflow trace ID")

    # audit
    audit_parser = subparsers.add_parser("audit", help="Audit full model lineage (training → registration → promotion)")
    audit_parser.add_argument("model", help="Model name/version (e.g. fraud-detector/3)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "verify":
        if args.verify_type == "run":
            sys.exit(cmd_verify_run(args))
        elif args.verify_type == "model":
            sys.exit(cmd_verify_model(args))
        elif args.verify_type == "trace":
            sys.exit(cmd_verify_trace(args))
        else:
            # Print help for the verify subparser by re-parsing.
            parser.parse_args(["verify", "--help"])
    elif args.command == "audit":
        sys.exit(cmd_audit(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
