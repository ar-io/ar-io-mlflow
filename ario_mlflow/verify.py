"""Four-check verification helpers for the pure-commitment design.

The four checks (see plan Part 3 verification flow):

1. **Cryptographic signature.** Validate Ed25519 signature on the
   envelope. Local, instant.
2. **Anchored bytes intact.** Download ``ario/payload.json`` from MLflow,
   re-hash, compare to the envelope's ``payload_hash``. Catches
   tampering with the canonical witness in MLflow's artifact store.
3. **Live MLflow matches anchored bytes.** Re-fetch the "live" fields
   (params, metrics, artifact checksums) from MLflow and rebuild the
   canonical payload, holding "snapshot" fields (caller metadata,
   trace IDs) constant from the original. Compare bytes. Catches
   tampering with MLflow's tracking-store data after anchoring.
4. **(Optional) ar.io Verify Level 3 attestation.** Independent
   third-party confirmation that the Arweave TX exists and is
   permanently stored. Implemented by ``ArioVerifyClient`` — unchanged
   from v1.

Verifiers wanting only signature + envelope-internal consistency can
call :func:`verify_signature` alone. Auditors with MLflow access run
:func:`full_verify` to get all four. The legacy three-level helpers
(``verify_record``, ``verify_arweave``, ``verify_ario``,
``full_verify`` in their v1 form) are deleted in this redesign — they
covered a different design where the proof carried the source data.
"""

import json
import logging
import os
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data, normalize_floats
from ario_mlflow.arweave import (
    ArweaveAnchor,
    _DEFAULT_MAX_RETRIES,
    _DEFAULT_RETRY_BACKOFF,
    _RETRY_STATUS_CODES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Live-field re-derivation per event type
# ---------------------------------------------------------------------------
#
# For check 3 we need to know which fields in the canonical payload are
# "live" (re-fetchable from MLflow at verify time) vs. "snapshot" (set at
# anchor time and unchanged thereafter). The hash of the rebuilt payload
# equals the original only if every live field still matches; any
# tampered live field flips check 3 to FAIL.
#
# Snapshot fields (taken from the downloaded payload.json as-is): event_id,
# event_type, signed_at, caller metadata (otel_trace_id, service_name,
# etc.), mlflow_trace_id (per-call, can't be re-derived from the run).


class LiveRefetchError(Exception):
    """Raised when a live-field refetcher can't obtain all expected fields.

    Treated as a check-3 FAIL by ``verify_source_of_truth`` — silently
    falling back to anchored values would let check 3 falsely pass for
    a tampered field whose refetch happens to fail. The caller knows
    something genuinely couldn't be verified rather than seeing a green
    check.
    """


def _fetch_trace_tags(mlflow_client, trace_id):
    """Fetch a trace's metadata + tags without loading its spans.

    Background: the prediction refetcher only needs the trace's tags
    (specifically ``ario.payload_json``) to re-derive the canonical
    bytes for verify-check 3. The full ``Trace`` object — fetched via
    ``mlflow_client.get_trace(trace_id)`` — additionally loads the
    trace's spans from its artifact repository, which on MLflow 3.x
    requires ``mlflow.artifactLocation`` to be present in the trace's
    tags. Some MLflow 3.x backend modes don't set that tag, and the
    spans-load step then raises ``Unable to determine trace artifact
    location`` — even though our caller never touches the spans.

    This helper prefers the lighter ``_tracing_client.get_trace_info``
    path (returns a ``TraceInfo`` with tags, no spans → no artifact
    lookup) and falls back to the public ``get_trace`` for older
    MLflow versions where the private tracing client isn't accessible.
    The downstream code in ``_refetch_prediction_live_fields`` handles
    both shapes (``Trace.info.tags`` and ``TraceInfo.tags``).
    """
    tracing_client = getattr(mlflow_client, "_tracing_client", None)
    if tracing_client is not None and hasattr(tracing_client, "get_trace_info"):
        return tracing_client.get_trace_info(trace_id)
    return mlflow_client.get_trace(trace_id)


# Per-event-type contracts for what fields the refetcher MUST produce.
# A refetcher that can't produce all of these is treated as a failure
# (raises LiveRefetchError); the silent-skip behaviour from before
# allowed tampered fields to pass check 3 when their refetch failed.
_TRAINING_REQUIRED_LIVE_FIELDS = frozenset({
    "params", "metrics", "artifact_checksums", "source_name", "git_commit",
})
# dataset_inputs is required only when the anchored payload commits to it
# (i.e. proofs anchored under v1+ of input-side anchoring). Legacy proofs
# from before input-side anchoring shipped have no dataset_inputs field
# and are still valid; the refetcher conditionally adds the field below.
_REGISTRATION_REQUIRED_LIVE_FIELDS = frozenset({
    "artifact_hash", "artifact_verified",
})


def _refetch_training_live_fields(payload: dict, mlflow_client) -> dict:
    """Re-fetch the live training fields from MLflow's current state.

    Raises ``LiveRefetchError`` if any required field can't be obtained.
    Required fields are listed in ``_TRAINING_REQUIRED_LIVE_FIELDS``.
    """
    from ario_mlflow.anchoring import (
        artifact_checksums, ArtifactAccessError, _logged_model_paths,
        _serialize_dataset_inputs,
    )

    run_id = payload.get("run_id")
    if not run_id:
        raise LiveRefetchError("payload missing run_id; cannot refetch training fields")

    try:
        run = mlflow_client.get_run(run_id)
    except Exception as e:  # noqa: BLE001
        raise LiveRefetchError(f"could not fetch run {run_id}: {e}") from e

    # Resolve artifact path the same way anchor() did — from the run's
    # logged-model history. Necessary for non-default paths.
    logged_paths = list(dict.fromkeys(_logged_model_paths(run)))
    if len(logged_paths) == 1:
        artifact_path = logged_paths[0]
    elif len(logged_paths) > 1:
        # Multiple models logged: anchor() rejects this case at write
        # time, so this branch is only hit for proofs anchored under a
        # weaker guard. Fall back to "model" — the verifier compares
        # bytes, so a mismatch will still surface as ok=False.
        artifact_path = "model"
    else:
        artifact_path = "model"

    fresh: dict = {
        "params": dict(run.data.params),
        "metrics": normalize_floats(dict(run.data.metrics), precision=6),
        "source_name": run.data.tags.get("mlflow.source.name", ""),
        "git_commit": run.data.tags.get("mlflow.source.git.commit", ""),
    }
    # Re-derive dataset_inputs only when the anchored payload commits to
    # it. Proofs anchored under v1+ of input-side anchoring have the
    # field; legacy proofs (anchored before input-side shipped) don't,
    # and unconditionally adding it on the refetch side would cause
    # rebuilt canonical bytes to diverge from the anchored bytes,
    # spuriously failing verification of legacy proofs.
    #
    # For v1+ proofs, any post-anchor mutation in MLflow (changed
    # digest, added fraudulent extra input, removed input) makes the
    # rebuilt canonical bytes diverge from the anchored bytes,
    # flipping source_of_truth to FAIL.
    if "dataset_inputs" in payload:
        fresh["dataset_inputs"] = _serialize_dataset_inputs(run)
    try:
        fresh["artifact_checksums"] = artifact_checksums(run_id, artifact_path=artifact_path)
    except ArtifactAccessError as e:
        # Fail-loud rather than silently keeping the anchored value.
        # Otherwise a tampered artifact_checksums field whose refetch
        # happens to fail would let check 3 falsely pass.
        raise LiveRefetchError(
            f"could not re-hash artifacts for run {run_id}: {e}"
        ) from e

    missing = _TRAINING_REQUIRED_LIVE_FIELDS - set(fresh.keys())
    if missing:
        raise LiveRefetchError(
            f"refetcher did not produce required field(s): {sorted(missing)}"
        )
    return fresh


def _refetch_registration_live_fields(payload: dict, mlflow_client) -> dict:
    """Re-fetch the live registration fields from MLflow's current state.

    Raises ``LiveRefetchError`` if any required field can't be
    obtained.
    """
    from ario_mlflow.anchoring import artifact_checksums, ArtifactAccessError, parse_runs_uri

    source_run_id = payload.get("source_run_id")
    if not source_run_id:
        raise LiveRefetchError("payload missing source_run_id")

    try:
        run = mlflow_client.get_run(source_run_id)
    except Exception as e:  # noqa: BLE001
        raise LiveRefetchError(
            f"could not fetch source run {source_run_id}: {e}"
        ) from e

    expected_hash = run.data.tags.get("ario.artifact_hash")
    fresh: dict = {"artifact_hash": expected_hash}

    src_run_id, src_artifact_path = parse_runs_uri(payload.get("source"))
    artifact_path = src_artifact_path or "model"
    try:
        checksums = artifact_checksums(source_run_id, artifact_path=artifact_path)
    except ArtifactAccessError as e:
        raise LiveRefetchError(
            f"could not re-hash artifacts for source run {source_run_id}: {e}"
        ) from e

    if expected_hash is None:
        # Source run has no anchored artifact_hash tag — likely an
        # un-anchored or pre-v2 training run. We can't reproduce
        # artifact_verified honestly.
        raise LiveRefetchError(
            f"source run {source_run_id} has no ario.artifact_hash tag; "
            f"cannot derive artifact_verified"
        )

    computed = hash_data(canonical_json(checksums)) if checksums else None
    fresh["artifact_verified"] = computed == expected_hash

    missing = _REGISTRATION_REQUIRED_LIVE_FIELDS - set(fresh.keys())
    if missing:
        raise LiveRefetchError(
            f"refetcher did not produce required field(s): {sorted(missing)}"
        )
    return fresh


_PREDICTION_REQUIRED_LIVE_FIELDS = frozenset({"_payload"})


def _refetch_prediction_live_fields(payload: dict, mlflow_client) -> dict:
    """Re-fetch the live prediction payload from the MLflow trace.

    Predictions don't have a re-derivable run.data surface like training
    does. Instead, ``VerifiedModel.predict`` mirrors the full canonical
    payload onto the MLflow trace as ``ario.payload_json``. This refetcher
    reads that tag and returns the parsed dict so
    :func:`verify_source_of_truth` can compare it to the anchored
    ``ario/predictions/<id>/payload.json`` artifact.

    The check catches MLflow-side tampering of the trace (someone modifying
    ``ario.payload_json`` without touching the artifact, or vice versa).
    Pruned traces are not a verification failure — the trace might be
    gone for retention reasons; we surface this via ``LiveRefetchError``
    which becomes ``ok=False, reason="live_refetch_incomplete"`` so the
    caller can distinguish "trace gone" from "data tampered."

    Raises ``LiveRefetchError`` when the trace can't be fetched, the tag
    is missing, or the JSON can't be parsed.
    """
    import mlflow

    trace_id = payload.get("mlflow_trace_id")
    if not trace_id:
        raise LiveRefetchError(
            "payload missing mlflow_trace_id; cannot refetch prediction live fields"
        )

    try:
        trace = _fetch_trace_tags(mlflow_client, trace_id)
    except Exception as e:  # noqa: BLE001 — wraps both the lite + fallback paths; LiveRefetchError surfaces as live_refetch_incomplete to the caller
        raise LiveRefetchError(
            f"could not fetch trace {trace_id}: {e}"
        ) from e

    # Tags live under .info.tags on Trace objects (returned by get_trace)
    # and directly under .tags on TraceInfo objects (returned by
    # get_trace_info — our preferred path on MLflow 3.x). Handle both.
    tags = {}
    info = getattr(trace, "info", None)
    if info is not None and getattr(info, "tags", None):
        tags = dict(info.tags)
    elif getattr(trace, "tags", None):
        tags = dict(trace.tags)

    payload_json_tag = tags.get("ario.payload_json")
    if not payload_json_tag:
        raise LiveRefetchError(
            f"trace {trace_id} missing ario.payload_json tag (may have been pruned)"
        )

    try:
        live_payload = json.loads(payload_json_tag)
    except (ValueError, TypeError) as e:
        raise LiveRefetchError(
            f"could not parse ario.payload_json from trace {trace_id}: {e}"
        ) from e

    # Fail closed if the parsed JSON isn't an object: verify_source_of_truth
    # later does ``rebuilt.update(fresh_fields)`` and ``fresh_fields.keys()``
    # which crash on lists / strings / numbers. Those would surface as a
    # cryptic verification error instead of a clean "trace tag is malformed".
    if not isinstance(live_payload, dict):
        raise LiveRefetchError(
            f"trace {trace_id} has non-object ario.payload_json "
            f"(got {type(live_payload).__name__}); expected a JSON object"
        )

    # Return the parsed payload as the "fresh" overlay. verify_source_of_truth
    # will rebuild = original | fresh and re-canonicalize. If the trace agrees
    # with the artifact, rebuilt_bytes == payload_bytes → ok=True.
    return live_payload


_LIVE_FIELD_REFETCHERS = {
    "training_complete": _refetch_training_live_fields,
    "model_registered": _refetch_registration_live_fields,
    "prediction": _refetch_prediction_live_fields,
}


# ---------------------------------------------------------------------------
# Payload-download helpers per subject type
# ---------------------------------------------------------------------------

# Subject types that the v2 design REQUIRES to have a persisted
# payload.json artifact. For these, a missing artifact is a verification
# FAILURE, not a "not applicable." Legacy v1 subject types
# (mlflow_trace, mlflow_decision) are tracked separately so they
# correctly degrade to "not applicable" rather than failing.
_SUBJECT_TYPES_WITH_REQUIRED_ARTIFACT = frozenset({
    "mlflow_run",            # training
    "mlflow_model_version",  # registration / promotion
    "mlflow_prediction",     # v2 predictions
})

_LEGACY_PREDICTION_SUBJECT_TYPES = frozenset({
    "mlflow_trace",
    "mlflow_decision",
})

# Subject types whose v1 design intentionally skips MLflow-side checks.
# The proof's value lives in the signature + Arweave attestation; live
# re-derivation against MLflow's registry is deferred to a follow-up
# (see standalone-dataset-anchoring plan, "Out of scope: cross-run
# dataset reuse / dedup" and SoT for dataset events).
#
# These return ok=None without a warning log — they're known and
# handled, not unknown subject types.
_DEFERRED_MLFLOW_CHECK_SUBJECT_TYPES = frozenset({
    "mlflow_dataset",
})


def _download_payload_for_envelope(
    envelope: dict, mlflow_client
) -> tuple[bytes | None, bool, str | None]:
    """Download the ``ario/payload.json`` artifact for an envelope's subject.

    Returns a tuple ``(payload_bytes, artifact_expected, reason)``:

    - ``payload_bytes``: the raw bytes on success, ``None`` on miss.
    - ``artifact_expected``: ``True`` if the v2 design requires this
      envelope to have a persisted payload artifact (a miss should fail
      check 2), ``False`` for legacy subject types where no artifact
      ever existed (a miss is "not applicable" / informational).
    - ``reason``: short string describing why bytes are absent when they
      are; ``None`` on success.

    Returning the raw bytes (vs. parsed JSON) lets the caller hash them
    directly without risking re-canonicalization drift.
    """
    import mlflow

    subject = envelope.get("subject", {})
    subject_type = subject.get("type")
    event_type = envelope.get("event_type")
    expected = subject_type in _SUBJECT_TYPES_WITH_REQUIRED_ARTIFACT

    # Resolve (run_id, artifact_path) per subject shape.
    run_id: str | None = None
    artifact_path: str | None = None

    if subject_type == "mlflow_run":
        run_id = subject.get("run_id")
        artifact_path = "ario/payload.json"
    elif subject_type == "mlflow_model_version":
        name = subject.get("name")
        version = subject.get("version")
        if not name or not version:
            return None, expected, "subject_missing_name_or_version"
        try:
            mv = mlflow_client.get_model_version(name, str(version))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not resolve {name}/v{version} during verify: {e}")
            return None, expected, f"could_not_resolve_model_version: {e}"
        run_id = mv.run_id
        if event_type == "model_registered":
            artifact_path = "ario/registration_payload.json"
        elif event_type == "stage_transition":
            # Promotion artifacts are now keyed by event_id (not version)
            # to avoid overwrite when a model version is promoted
            # multiple times. Subject must carry event_id; envelope's
            # event_id field is the fallback.
            event_id = subject.get("event_id") or envelope.get("event_id")
            if not event_id:
                return None, expected, "subject_missing_event_id"
            artifact_path = f"ario/promotions/{event_id}/payload.json"
        else:
            artifact_path = "ario/payload.json"
    elif subject_type == "mlflow_prediction":
        decision_id = subject.get("decision_id")
        run_id = subject.get("model_run_id")
        if not decision_id or not run_id:
            return None, expected, "subject_missing_decision_id_or_run_id"
        artifact_path = f"ario/predictions/{decision_id}/payload.json"
    elif subject_type in _LEGACY_PREDICTION_SUBJECT_TYPES:
        # v1 predictions had no payload artifact. Caller should treat
        # this as "not applicable", not "failure."
        return None, False, "legacy_subject_type_no_artifact"
    elif subject_type in _DEFERRED_MLFLOW_CHECK_SUBJECT_TYPES:
        # Standalone dataset events (and similar) skip MLflow-side
        # checks in v1. Their value comes from signature + ar.io
        # attestation; live re-derivation is a follow-up.
        return None, False, "mlflow_check_deferred_for_subject"
    else:
        logger.warning(f"Unknown subject type for download: {subject_type!r}")
        return None, expected, f"unknown_subject_type: {subject_type!r}"

    if not run_id:
        return None, expected, "subject_missing_run_id"

    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact_path,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"Could not download {artifact_path} for run {run_id} during verify: {e}"
        )
        return None, expected, f"download_failed: {e}"

    try:
        with open(local_path, "rb") as f:
            return f.read(), expected, None
    except OSError as e:
        logger.warning(f"Could not read downloaded payload at {local_path}: {e}")
        return None, expected, f"read_failed: {e}"


# ---------------------------------------------------------------------------
# The four checks
# ---------------------------------------------------------------------------

def verify_signature(envelope: dict, proof_engine: ProofEngine) -> dict:
    """Check 1: cryptographic signature on the envelope.

    Wraps :meth:`ProofEngine.verify_commitment` (without payload bytes —
    that's check 2). Returns ``{ok: bool, ...}`` for uniform composition
    with the other checks.

    An envelope advertising an unsupported ``spec_version`` major fails
    with ``ok=False, reason="unsupported_spec_version"`` regardless of
    whether the signature itself is well-formed — the verifier doesn't
    know what shape the envelope is supposed to have, so it can't
    meaningfully attest to it. Legacy envelopes (no ``spec_version``
    field, anchored before this build) verify normally; callers see
    ``legacy_envelope=True`` so they can surface that distinction.
    """
    result = proof_engine.verify_commitment(envelope)
    spec_status = result.get("spec_version_status", "supported")
    legacy = result.get("legacy_envelope", False)
    sig_valid = result["signature_valid"]

    out = {
        "ok": sig_valid and spec_status != "unsupported",
        "signature_valid": sig_valid,
        "spec_version_status": spec_status,
        "legacy_envelope": legacy,
    }
    if spec_status == "unsupported":
        out["reason"] = "unsupported_spec_version"
    return out


def verify_anchored_bytes(envelope: dict, mlflow_client) -> dict:
    """Check 2: anchored bytes intact.

    Downloads ``ario/payload.json`` from MLflow, re-hashes the bytes,
    compares to ``envelope["payload_hash"]``. Catches tampering with
    the canonical witness in MLflow's artifact store.

    Result semantics:

    - ``ok=True`` — artifact downloaded, hash matches.
    - ``ok=False`` — either the hash doesn't match (tampering) **or**
      the artifact is missing for an event type that's required to
      have one (training, registration, promotion, v2 prediction).
      Missing-when-required is treated as a failure because that's
      precisely the regression check 2 is meant to catch — without
      this, a wiped artifact would silently pass.
    - ``ok=None`` — only for legacy v1 subject types
      (``mlflow_trace`` / ``mlflow_decision``) where no payload
      artifact ever existed. Here ``None`` is honest "not applicable",
      not "couldn't verify."
    """
    payload_bytes, artifact_expected, reason = _download_payload_for_envelope(
        envelope, mlflow_client,
    )
    stored = envelope.get("payload_hash")

    if payload_bytes is None:
        # Differentiate "expected but missing" (FAILURE) from "not
        # expected for this subject type" (legitimate not-applicable).
        return {
            "ok": False if artifact_expected else None,
            "reason": reason or "payload_artifact_not_available",
            "computed_hash": None,
            "stored_hash": stored,
            "payload_bytes": None,
            "artifact_expected": artifact_expected,
        }

    computed = hash_data(payload_bytes)
    return {
        "ok": computed == stored,
        "computed_hash": computed,
        "stored_hash": stored,
        "payload_bytes": payload_bytes,
        "artifact_expected": artifact_expected,
    }


def verify_source_of_truth(
    envelope: dict,
    payload_bytes: bytes,
    mlflow_client,
) -> dict:
    """Check 3: live MLflow data still matches anchored bytes.

    Parses ``payload_bytes`` (from check 2's download), re-fetches the
    "live" fields from MLflow's current state, and compares the rebuilt
    canonical bytes to the original. Catches MLflow-side tampering after
    anchoring — the central guarantee of the redesign.

    Args:
        envelope: The signed envelope (for event_type / subject).
        payload_bytes: The canonical bytes downloaded by check 2. Must
            be the exact bytes — re-canonicalizing parsed JSON could
            produce different output if the parsed dict's iteration
            order differs.
        mlflow_client: An ``MlflowClient`` for live re-fetching.
    """
    if not payload_bytes:
        return {"ok": None, "reason": "no_payload_to_compare"}

    event_type = envelope.get("event_type")
    refetcher = _LIVE_FIELD_REFETCHERS.get(event_type)
    if refetcher is None:
        return {
            "ok": None,
            "reason": "no_live_fields_for_event_type",
            "event_type": event_type,
            "note": (
                "Predictions commit to hashes of input/output. To run check 3 "
                "for a prediction, hash your copy of the raw input/output and "
                "compare to the payload's input_hash / output_hash directly."
            ),
        }

    try:
        original_payload = json.loads(payload_bytes)
    except (ValueError, TypeError) as e:
        return {"ok": False, "reason": f"payload_parse_failed: {e}"}

    try:
        fresh_fields = refetcher(original_payload, mlflow_client)
    except LiveRefetchError as e:
        # The refetcher raises when it can't completely refetch — that
        # means we can't confirm the live MLflow state, full stop.
        # Treating this as ok=False prevents the silent-fall-through
        # bug where a tampered field whose refetch fails would still
        # let check 3 pass against the anchored value.
        return {"ok": False, "reason": "live_refetch_incomplete", "detail": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"live_refetch_failed: {e}"}

    # Build the rebuilt payload: original + live overrides. Snapshot
    # fields (anything not in fresh_fields) flow through unchanged.
    rebuilt = dict(original_payload)
    rebuilt.update(fresh_fields)
    rebuilt_bytes = canonical_json(rebuilt)

    return {
        "ok": rebuilt_bytes == payload_bytes,
        "rebuilt_bytes": rebuilt_bytes,
        "live_fields_refetched": sorted(fresh_fields.keys()),
    }


# Default minimum ar.io Verify attestation level for ``ok=True``.
#
# ar.io Verify reports the maturity of an Arweave transaction as it
# settles through the network:
# - Level 1: indexed, not yet validated. Common for fresh TXs (seconds
#   to minutes after anchor).
# - Level 2: at least one gateway has downloaded and validated the
#   data + signature. Reached within minutes for most TXs.
# - Level 3: multi-gateway consensus. Strongest assurance; can take
#   hours or longer to reach.
#
# Default of 2 balances "false-pass risk" (Level 1 alone is a weak
# signal — TX appears in the index but no one has validated it) against
# "false-fail risk" (Level 3 only would fail every fresh anchor for
# hours). Audit / compliance use cases should pass
# ``min_attestation_level=3`` for stricter checks. Continuous-verification
# tooling that runs immediately after anchor can pass 1 to accept
# "indexed" as good enough.
#
# See ROADMAP "Receipts vs. attestation as a two-stage verify UX" for
# the longer-term refinement that surfaces this maturity explicitly
# rather than as a single pass/fail bar.
DEFAULT_MIN_ATTESTATION_LEVEL = 2


def verify_ario_attestation(
    envelope: dict,
    ario_client: "ArioVerifyClient | None",
    *,
    min_attestation_level: int = DEFAULT_MIN_ATTESTATION_LEVEL,
) -> dict:
    """Check 4 (optional): ar.io Verify attestation level.

    Calls the ar.io Verify service to confirm the Arweave TX exists and
    is being attested. Returns the attestation result (``level``,
    ``attested_by``, ``report_url``, ...) regardless of pass/fail —
    callers can surface the level explicitly to differentiate
    "passing strict bar" from "below threshold but maturing."

    Args:
        envelope: The signed envelope. Must carry ``_tx_id`` or
            ``arweave_tx_id`` so the call can be routed.
        ario_client: An ``ArioVerifyClient``. ``None`` or disabled
            client returns ``ok=None`` ("not applicable" / "not
            checked").
        min_attestation_level: Threshold below which ``ok=False``.
            Defaults to ``DEFAULT_MIN_ATTESTATION_LEVEL`` (2). Pass
            ``3`` for strict audit. Pass ``1`` to accept "TX is at
            least indexed."

    Returns:
        Dict with ``ok``, ``attestation_level``, ``attested_by``,
        ``attested_at``, ``report_url``, ``pdf_url``,
        ``min_attestation_level`` (the threshold used), and ``reason``
        on failure.
    """
    if ario_client is None or not getattr(ario_client, "enabled", False):
        return {"ok": None, "reason": "ario_verify_not_enabled"}

    # Pull the Arweave TX ID from wherever it's stored. The envelope
    # itself doesn't carry the TX (the TX is the address ON Arweave) —
    # callers passing an envelope here are expected to also know the
    # TX, typically from MLflow tags (ario.training_tx, etc.). For now
    # we accept it via a special key the caller adds; future API may
    # split the envelope and the TX more cleanly.
    tx_id = envelope.get("_tx_id") or envelope.get("arweave_tx_id")
    if not tx_id:
        return {"ok": None, "reason": "no_tx_id_provided"}

    result = ario_client.submit_verification(tx_id)
    if not result:
        return {"ok": False, "reason": "ario_verify_returned_no_result"}

    level = result.get("attestation_level") or 0
    base = {
        "attestation_level": level,
        "attested_by": result.get("attested_by"),
        "attested_at": result.get("attested_at"),
        "report_url": result.get("report_url"),
        "pdf_url": result.get("pdf_url"),
        "min_attestation_level": min_attestation_level,
    }

    if level >= min_attestation_level:
        return {"ok": True, **base}
    return {
        "ok": False,
        "reason": "attestation_level_below_threshold",
        **base,
    }


# Event types whose v2 contract requires both check 2 (anchored bytes
# intact) AND check 3 (live MLflow matches anchored payload) to pass for
# overall_ok=True. ok=None on either of these for these event types is a
# FAIL — silent neutrality would hide regressions check 2/3 are meant to
# catch.
#
# Predictions are included because ``VerifiedModel.predict`` mirrors the
# canonical payload onto the trace as ``ario.payload_json``, giving us
# a real second MLflow surface to compare against the artifact (parallel
# to training's params/metrics surface). A pruned trace surfaces as
# ``ok=False, reason="live_refetch_incomplete"`` — the auditor sees a
# clear "trace not available" rather than a silent pass.
_REQUIRES_FULL_MLFLOW_VERIFICATION = frozenset({
    "training_complete",
    "model_registered",
    "prediction",
})


def _compute_overall_ok(envelope: dict, sig: dict, bytes_check: dict, sot: dict, ario: dict) -> bool | None:
    """Combine the four check results into an overall pass/fail.

    Rules:
    - If any check is explicitly ``False``: overall=False.
    - For event types in ``_REQUIRES_FULL_MLFLOW_VERIFICATION``,
      ``ok=None`` on signature / anchored_bytes / source_of_truth is
      treated as a FAIL — these checks are required, and a None result
      means "we couldn't actually verify this." Silent neutrality here
      would hide exactly the regressions the redesign was built to catch.
    - For other event types (predictions, legacy subjects), ``ok=None``
      stays neutral — these have weaker verification contracts where a
      missing artifact is genuinely "not applicable."
    - If at least one True and no False/required-None: overall=True.
    - Otherwise: overall=None (nothing meaningful was checked).
    """
    event_type = envelope.get("event_type")
    requires_full = event_type in _REQUIRES_FULL_MLFLOW_VERIFICATION

    statuses = [sig["ok"], bytes_check["ok"], sot["ok"], ario["ok"]]

    # Hard fail on any explicit False.
    if any(s is False for s in statuses):
        return False

    # For envelope types with a strict v2 contract, ok=None on a
    # required check (signature, anchored bytes, source of truth) is
    # also a fail. ar.io is genuinely optional — None is acceptable.
    if requires_full:
        required_checks = [sig["ok"], bytes_check["ok"], sot["ok"]]
        if any(s is None for s in required_checks):
            return False

    # If at least one explicit True, overall is True.
    if any(s is True for s in statuses):
        return True

    # Nothing meaningful checked.
    return None


def full_verify(
    envelope: dict,
    *,
    proof_engine: ProofEngine,
    mlflow_client=None,
    ario_client: "ArioVerifyClient | None" = None,
    min_attestation_level: int = DEFAULT_MIN_ATTESTATION_LEVEL,
) -> dict:
    """Run all four checks and return a combined result.

    Each check is independent — failures in one don't short-circuit the
    others, so the caller sees the complete state. ``overall`` follows
    the rules in :func:`_compute_overall_ok`: training and registration
    envelopes require all three local checks to pass; other event types
    are more permissive about None results.
    """
    sig = verify_signature(envelope, proof_engine)
    bytes_check = (
        verify_anchored_bytes(envelope, mlflow_client) if mlflow_client else
        {"ok": None, "reason": "no_mlflow_client"}
    )
    sot = (
        verify_source_of_truth(envelope, bytes_check.get("payload_bytes") or b"", mlflow_client)
        if mlflow_client and bytes_check.get("payload_bytes")
        else {"ok": None, "reason": "no_payload_to_compare"}
    )
    ario = (
        verify_ario_attestation(envelope, ario_client, min_attestation_level=min_attestation_level)
        if ario_client else {"ok": None, "reason": "no_ario_client"}
    )

    overall = _compute_overall_ok(envelope, sig, bytes_check, sot, ario)

    return {
        "signature": sig,
        "anchored_bytes": bytes_check,
        "source_of_truth": sot,
        "ario_attestation": ario,
        "overall": overall,
    }


# ---------------------------------------------------------------------------
# Auditor-shaped primitive: verify_record + verify_proof_by_tx wrapper
# ---------------------------------------------------------------------------
#
# verify_record is the foundation for the auditor flow. Given an envelope
# plus the canonical bytes that hash to its payload_hash, it runs:
#   - signature check (envelope-internal, offline)
#   - anchored-bytes check (re-hash provided canonical bytes vs payload_hash)
#   - optional ar.io attestation (network call to ar.io Verify gateway)
#
# It deliberately does NOT do check 3 (source-of-truth re-derivation from
# live MLflow). For an auditor working from a portable bundle, the bundle
# IS the source of truth — there is no operator-side MLflow to reach for.
# Operator-side flows that DO have MLflow access call verify_proof_by_tx,
# which composes verify_record with the additional SoT step + the
# Arweave-fetch step.

def _verify_canonical_bytes_match(envelope: dict, canonical_bytes: bytes) -> dict:
    """Re-hash provided canonical bytes against the envelope's payload_hash.

    Auditor-shaped variant of :func:`verify_anchored_bytes` — same logical
    check but takes bytes already in hand instead of downloading them
    from MLflow. Returns the same shape (``ok``, ``computed_hash``,
    ``stored_hash``) so callers can treat both paths uniformly.
    """
    stored = envelope.get("payload_hash")
    if not canonical_bytes:
        return {
            "ok": False,
            "reason": "no_canonical_bytes_provided",
            "computed_hash": None,
            "stored_hash": stored,
        }
    computed = hash_data(canonical_bytes)
    return {
        "ok": computed == stored,
        "computed_hash": computed,
        "stored_hash": stored,
    }


def verify_record(
    envelope: dict,
    canonical_bytes: bytes,
    *,
    proof_engine: ProofEngine,
    ario_client: "ArioVerifyClient | None" = None,
    min_attestation_level: int = DEFAULT_MIN_ATTESTATION_LEVEL,
) -> dict:
    """Verify a record from a portable bundle (auditor-shaped primitive).

    Runs the three checks an auditor can perform without MLflow access:

    1. Signature on the envelope (offline; envelope-internal).
    2. Anchored-bytes match: hash ``canonical_bytes`` and compare to
       ``envelope["payload_hash"]``.
    3. (Optional) ar.io attestation that the proof exists on Arweave —
       requires the envelope to carry ``_tx_id`` and a live ``ario_client``.

    The "live MLflow re-derivation" check (check 3 in the operator flow)
    is intentionally omitted: an auditor working from a portable bundle
    treats the bundle itself as the source of truth. Operator-side flows
    use :func:`verify_proof_by_tx` to add that check.

    Trusted-issuer enforcement (require the embedded signing key to
    belong to a known set) is a follow-up task and not yet wired —
    when it lands it'll be added as a ``trusted_issuer_keys`` keyword
    parameter at the same time as the underlying ``verify_signature``
    plumbing. Adding the parameter here today would create a no-op
    security knob, so it's deferred.

    Args:
        envelope: Signed envelope dict (the proof itself).
        canonical_bytes: The exact canonical-JSON bytes that were hashed
            to produce ``envelope["payload_hash"]``. Must be the original
            bytes — re-canonicalizing parsed JSON could produce drift.
        proof_engine: A ``ProofEngine`` for signature verification.
        ario_client: Optional ``ArioVerifyClient``. When ``None`` or
            disabled, the ar.io attestation check returns ``ok=None``
            ("not checked") rather than ``ok=False``.
        min_attestation_level: Threshold below which ar.io attestation
            returns ``ok=False``. See :data:`DEFAULT_MIN_ATTESTATION_LEVEL`.

    Returns:
        Dict with ``signature``, ``anchored_bytes``, ``ario_attestation``,
        and ``overall``. ``overall`` is True when sig and bytes are both
        True and ar.io is not False (None counts as "not checked, not
        failed" because the auditor explicitly chose to skip).
    """
    sig = verify_signature(envelope, proof_engine)

    bytes_check = _verify_canonical_bytes_match(envelope, canonical_bytes)

    ario = (
        verify_ario_attestation(envelope, ario_client, min_attestation_level=min_attestation_level)
        if ario_client else {"ok": None, "reason": "no_ario_client"}
    )

    # Auditor-flow overall semantics:
    # - sig and bytes must both be True.
    # - ar.io is True (verified) or None (not checked) — both acceptable.
    # - Any explicit False fails overall.
    if any(c.get("ok") is False for c in (sig, bytes_check, ario)):
        overall: bool | None = False
    elif sig.get("ok") is True and bytes_check.get("ok") is True:
        overall = True
    else:
        overall = None

    return {
        "signature": sig,
        "anchored_bytes": bytes_check,
        "ario_attestation": ario,
        "overall": overall,
    }


def verify_proof_by_tx(
    tx_id: str,
    *,
    anchor: ArweaveAnchor,
    proof_engine: ProofEngine,
    mlflow_client=None,
    ario_client: "ArioVerifyClient | None" = None,
    min_attestation_level: int = DEFAULT_MIN_ATTESTATION_LEVEL,
) -> dict:
    """Fetch envelope from Arweave by TX, then run all four operator-side checks.

    The operator-side counterpart to :func:`verify_record`. When the
    caller has only a TX ID (typical in the demo: the UI knows the TX
    from MLflow tags but doesn't have the envelope in memory), this
    helper handles the fetch and assembles the full four-check result.

    Adds a top-level ``proof_found`` flag so consumers can distinguish
    "envelope retrieved from Arweave" from "envelope was missing" — what
    the demo's "Proof Found" verification row is supposed to express.

    When the fetch fails, every sub-check returns ``ok=None`` (the
    checks weren't actually run). When the fetch succeeds, the four
    checks run in the same shape as :func:`full_verify`.

    Trusted-issuer enforcement is a follow-up task; see
    :func:`verify_record` for the rationale on why a no-op
    ``trusted_issuer_keys`` parameter isn't accepted today.

    Args:
        tx_id: Arweave transaction ID for the proof.
        anchor: An ``ArweaveAnchor`` (or compatible object exposing
            ``fetch_proof(tx_id)``) for retrieval.
        proof_engine: A ``ProofEngine`` for signature verification.
        mlflow_client: Optional ``MlflowClient``. Required for checks 2
            and 3; when omitted those return ``ok=None``.
        ario_client: Optional ``ArioVerifyClient``. When omitted, ar.io
            attestation returns ``ok=None``.
        min_attestation_level: Threshold for ar.io attestation. See
            :data:`DEFAULT_MIN_ATTESTATION_LEVEL`.

    Returns:
        Dict with ``proof_found``, ``signature``, ``anchored_bytes``,
        ``source_of_truth``, ``ario_attestation``, and ``overall``.
    """
    plugin_envelope = anchor.fetch_proof(tx_id)
    if plugin_envelope is None:
        return {
            "proof_found": False,
            "signature": {"ok": None, "reason": "no_envelope"},
            "anchored_bytes": {"ok": None, "reason": "no_envelope"},
            "source_of_truth": {"ok": None, "reason": "no_envelope"},
            "ario_attestation": {"ok": None, "reason": "no_envelope"},
            "overall": None,
        }

    # Caller-attached metadata. _tx_id flows through to the ar.io
    # attestation check; underscore-prefixed keys are stripped before
    # signature canonicalization (see test_verify_commitment_ignores_
    # underscore_prefixed_caller_annotations) so this doesn't break
    # check 1.
    plugin_envelope["_tx_id"] = tx_id

    result = full_verify(
        plugin_envelope,
        proof_engine=proof_engine,
        mlflow_client=mlflow_client,
        ario_client=ario_client,
        min_attestation_level=min_attestation_level,
    )
    result["proof_found"] = True
    return result


class ArioVerifyClient:
    """Client for AR.IO Verify REST API.

    Uses a shared :class:`requests.Session` with a retry policy
    (5xx + 429 with exponential backoff) so transient gateway failures
    don't show up as a hard "ar.io Verify failed" verdict.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff_factor: float = _DEFAULT_RETRY_BACKOFF,
        health_timeout: float = 5.0,
        submit_timeout: float = 30.0,
    ):
        self.base_url = (base_url or os.environ.get("ARIO_MLFLOW_ARIO_VERIFY_URL", "")).rstrip("/")
        self.enabled = False
        # Last failure surfaced to callers that get None from
        # submit_verification. Reset on each call.
        self.last_error: str | None = None
        self._submit_timeout = submit_timeout

        # Shared session with retry on transient failures.
        self._session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=retry_backoff_factor,
            status_forcelist=_RETRY_STATUS_CODES,
            allowed_methods=("GET", "POST"),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        if not self.base_url:
            return

        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=health_timeout)
            if resp.status_code == 200:
                self.enabled = True
                logger.info(f"ar.io Verify connected at {self.base_url}")
            else:
                logger.warning(
                    f"ar.io Verify health check returned HTTP {resp.status_code} "
                    f"at {self.base_url}; client disabled"
                )
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"ar.io Verify unavailable at {self.base_url}: {type(e).__name__}: {e}"
            )

    def submit_verification(self, tx_id: str) -> dict | None:
        self.last_error = None
        if not self.enabled:
            self.last_error = "ar.io Verify client not enabled"
            return None
        try:
            resp = self._session.post(
                f"{self.base_url}/api/v1/verify",
                json={"txId": tx_id},
                timeout=self._submit_timeout,
            )
            resp.raise_for_status()
            return self._normalize(resp.json())
        except requests.exceptions.RequestException as e:
            self.last_error = f"submit_verification network/HTTP error: {type(e).__name__}: {e}"
            logger.error(f"ar.io Verify failed for {tx_id}: {self.last_error}")
            return None
        except (ValueError, KeyError) as e:
            self.last_error = f"submit_verification response parse error: {type(e).__name__}: {e}"
            logger.error(f"ar.io Verify response invalid for {tx_id}: {self.last_error}")
            return None

    def poll_attestation(
        self,
        tx_id: str,
        *,
        target_level: int = 2,
        timeout: float = 120.0,
        interval: float = 5.0,
    ) -> dict | None:
        """Poll ``submit_verification`` until ``target_level`` is reached.

        ar.io Verify's ``attestation_level`` grows over time as a proof
        propagates: roughly 1 = anchored, 2 = content matches, 3 =
        signature attested by an operator. Callers wanting stronger
        maturity than the immediate result can use this helper instead
        of polling externally.

        Returns the latest verification dict — at ``target_level`` or
        higher on success, or whatever level was last seen if the
        timeout expired. Returns ``None`` only if the client is
        disabled or every submit attempt failed (in which case
        ``self.last_error`` carries the last failure).
        """
        if not self.enabled:
            self.last_error = "ar.io Verify client not enabled"
            return None

        deadline = time.monotonic() + timeout
        last_result: dict | None = None
        attempts = 0
        while True:
            attempts += 1
            result = self.submit_verification(tx_id)
            if result is not None:
                last_result = result
                level = result.get("attestation_level") or 0
                if level >= target_level:
                    logger.info(
                        f"ar.io Verify reached level {level} for {tx_id} "
                        f"after {attempts} attempt(s)"
                    )
                    return result
            if time.monotonic() >= deadline:
                if last_result is None:
                    logger.warning(
                        f"ar.io Verify polling for {tx_id} got no successful "
                        f"response in {timeout}s ({attempts} attempts)"
                    )
                else:
                    logger.info(
                        f"ar.io Verify polling for {tx_id} timed out at level "
                        f"{last_result.get('attestation_level')} after {attempts} attempt(s); "
                        f"target was {target_level}"
                    )
                return last_result
            time.sleep(interval)

    def _normalize(self, data: dict) -> dict:
        # ar.io Verify returns explicit nulls for sub-objects when no
        # attestation is yet available (e.g. for fresh TXs not yet
        # indexed). dict.get(key, {}) returns the key's actual value
        # when present — None, not the {} default. Use ``or {}`` to
        # collapse both None and missing into {}.
        links = data.get("links") or {}
        attestation = data.get("attestation") or {}
        existence = data.get("existence") or {}

        def resolve(path):
            if not path:
                return None
            return path if path.startswith("http") else f"{self.base_url}{path}"

        return {
            "verification_id": data.get("verificationId"),
            "status": existence.get("status", "unknown"),
            "attestation_level": data.get("level"),
            "report_url": resolve(links.get("dashboard")),
            "pdf_url": resolve(links.get("pdf")),
            "attested_by": attestation.get("gateway"),
            "attested_at": attestation.get("attestedAt"),
        }


# Note on deleted v1 helpers: the legacy three-level helpers
# (verify_record / verify_arweave / verify_ario / legacy full_verify)
# lived at the bottom of this module in v1. They covered the v1
# envelope shape where source data was inside the proof itself; the
# pure-commitment design needs different checks (download payload.json,
# compare hash, re-derive from MLflow). The new helpers near the top
# of this file (verify_signature / verify_anchored_bytes /
# verify_source_of_truth / verify_ario_attestation / new full_verify)
# are the replacements. CLI consumers updated in Phase 1.9.
