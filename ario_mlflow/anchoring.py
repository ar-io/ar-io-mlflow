"""Public anchor() API and artifact checksum utilities."""

import hashlib
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone

import mlflow

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data, normalize_floats
from ario_mlflow.arweave import ArweaveAnchor
from ario_mlflow.report import generate_verification_html

logger = logging.getLogger(__name__)


# Tag key for the per-registered-model chain head pointer. Each new training
# proof for a model reads this tag for its ``previous_hash`` and writes the
# new ``payload_hash`` back. See plan Part 3 design principle 5 and Part 4
# plugin change item 2 for the per-event-type chain semantics.
TAG_LAST_TRAINING_HASH = "ario.last_training_hash"


# OTel auto-capture opt-out env var. Set to "false" / "0" / "no" / "off"
# to disable auto-capture process-wide. Per-call opt-out is via the
# ``capture_otel=False`` parameter on ``anchor()``,
# ``VerifiedModel.predict()``, and ``ArioMlflowClient`` registration
# paths.
ENV_CAPTURE_OTEL = "ARIO_MLFLOW_CAPTURE_OTEL"


def capture_otel_context() -> dict:
    """Auto-capture OTel trace_id / span_id from an active span if present.

    Returns an empty dict when:
    - ``opentelemetry-api`` is not installed.
    - No active OTel span / invalid span context.
    - ``ARIO_MLFLOW_CAPTURE_OTEL`` env var is set to ``false`` / ``0`` /
      ``no`` / ``off``.

    Otherwise returns ``{"otel_trace_id": <32-hex>, "otel_span_id":
    <16-hex>}`` suitable for merging into the canonical payload. Caller-
    supplied ``metadata={"otel_trace_id": ...}`` wins over the auto-
    captured value (see ``_build_training_payload`` and the equivalent
    builders in ``client.py`` / ``model.py``).

    Why default-on: the plugin already auto-captures MLflow ``trace_id``
    when present; OTel is the more universally-adopted observability
    system. Default-off would silently produce proofs without
    cross-stack correlation in production deployments. Privacy-conscious
    deployments can opt out via env var or the ``capture_otel=False``
    parameter on the entry points. Soft-imports OTel — no hard
    dependency on ``opentelemetry-api``.
    """
    optout = os.environ.get(ENV_CAPTURE_OTEL, "").lower()
    if optout in ("false", "0", "no", "off"):
        return {}
    try:
        from opentelemetry import trace as _otel_trace  # noqa: PLC0415
    except ImportError:
        return {}
    try:
        ctx = _otel_trace.get_current_span().get_span_context()
    except Exception:  # noqa: BLE001
        return {}
    if not getattr(ctx, "is_valid", False):
        return {}
    return {
        "otel_trace_id": format(ctx.trace_id, "032x"),
        "otel_span_id": format(ctx.span_id, "016x"),
    }


class ArtifactAccessError(RuntimeError):
    """Raised when an MLflow run's artifacts cannot be downloaded or read for hashing.

    Callers must NOT treat this as "no artifacts" — the true state is unknown and
    they should skip writing an `ario.artifact_hash` rather than anchor a hash of
    an empty tree as if it were a real provenance record.
    """


def _logged_model_paths(run_data) -> list[str]:
    """Return the artifact paths of every model logged in this run.

    MLflow writes a ``mlflow.log-model.history`` tag whose value is a JSON list
    describing each ``mlflow.<flavor>.log_model`` call in the run. Reading this
    tag lets ``anchor()`` hash whatever the user actually logged, rather than
    silently defaulting to ``"model"`` and skipping the hash when the caller
    used a different name.
    """
    history_json = run_data.data.tags.get("mlflow.log-model.history")
    if not history_json:
        return []
    try:
        history = json.loads(history_json)
    except (ValueError, TypeError):
        return []
    paths = []
    for entry in history:
        if isinstance(entry, dict) and entry.get("artifact_path"):
            paths.append(entry["artifact_path"])
    return paths


def parse_runs_uri(source: str | None) -> tuple[str | None, str | None]:
    """Parse a ``runs:/<run_id>/<artifact_path>`` URI.

    Returns ``(run_id, artifact_path)`` where either element may be ``None`` if
    the source is missing, not a ``runs:/`` URI, or has no artifact path. This
    matters because MLflow's ``ModelVersion.source`` preserves the original
    artifact path from registration (e.g. ``sklearn-model``, ``keras-model``)
    and we must not assume it is always ``model``.
    """
    if not source or not source.startswith("runs:/"):
        return None, None
    rest = source[len("runs:/"):].lstrip("/")
    if "/" not in rest:
        return (rest or None), None
    run_id, artifact_path = rest.split("/", 1)
    return (run_id or None), (artifact_path or None)


# Files MLflow synthesizes / writes as part of its model-registry
# bookkeeping after a model version is registered. They are NOT part of
# the model content; they're pointers MLflow adds saying "this artifact
# was registered as version X." Excluding them from the integrity hash
# is essential — otherwise a model anchored *before* registration (the
# normal anchor()-then-create_model_version order) would re-hash to a
# different value once registration adds these files, falsely tripping
# VerifiedModel's IntegrityError.
#
# If MLflow adds new bookkeeping filenames in future versions, add them
# here. The contract: "registered_model_meta and friends are not part
# of the model's signed content."
_MLFLOW_REGISTRATION_METADATA_FILES = frozenset({
    "registered_model_meta",
})


def artifact_checksums(client_or_run_id, run_id: str | None = None, artifact_path: str = "model") -> dict[str, str]:
    """Compute SHA-256 checksums of model artifacts in an MLflow run.

    Uses ``mlflow.artifacts.download_artifacts`` which works with both
    file-based and database-backed tracking stores in MLflow 3.x.

    Excludes MLflow's post-registration bookkeeping files (see
    ``_MLFLOW_REGISTRATION_METADATA_FILES``) so the hash is stable
    across the anchor→register lifecycle. The model's actual content
    files (``MLmodel``, ``model.pkl``, ``conda.yaml``, etc.) are
    hashed normally.

    Args:
        client_or_run_id: An MlflowClient (ignored, kept for backward compat) or a run_id string.
        run_id: The run ID. If client_or_run_id is a string, this is ignored.
        artifact_path: Artifact subdirectory to hash (default "model").
    """
    if isinstance(client_or_run_id, str):
        run_id = client_or_run_id
    try:
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    except Exception as e:  # noqa: BLE001 — wraps any MLflow artifact-access failure as ArtifactAccessError so callers can decide; not a silent skip
        # Callers must not silently anchor an empty tree as if it were the
        # artifact's real hash — surface the failure so they can skip.
        raise ArtifactAccessError(
            f"Could not download artifacts for run {run_id!r} at path {artifact_path!r}: {e}"
        ) from e

    checksums: dict[str, str] = {}
    for root, _dirs, files in os.walk(local_path):
        for fname in files:
            if fname in _MLFLOW_REGISTRATION_METADATA_FILES:
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, local_path)
            try:
                with open(fpath, "rb") as f:
                    checksums[rel] = hashlib.sha256(f.read()).hexdigest()
            except OSError as e:
                raise ArtifactAccessError(
                    f"Failed to read artifact file {fpath!r} for run {run_id!r}: {e}"
                ) from e
    return checksums


def _find_registered_model_for_run(client, run_id: str) -> str | None:
    """Find the registered model name (if any) whose source is this run.

    Used to locate the per-model chain-head tag (``ario.last_training_hash``)
    so a new training proof can chain to the previous training of the same
    model. Returns the first match's name, or ``None`` if no registered
    model points at this run.

    A run can in principle have multiple registered models (e.g. logged
    several models then registered each). For the chain-head linkage we
    use the first match deterministically; truly multi-model training
    runs are uncommon and the audit chain still works (each registration
    chains to ``ario.training_tx`` on the run).
    """
    try:
        results = client.search_model_versions(f"run_id='{run_id}'")
    except Exception as e:  # noqa: BLE001
        logger.debug(
            f"search_model_versions for run {run_id} failed; "
            f"chain-head update will be skipped: {e}"
        )
        return None
    if not results:
        return None
    return results[0].name


def _serialize_dataset_inputs(run_data) -> list[dict]:
    """Read ``run_data.inputs.dataset_inputs`` and produce a
    deterministic list suitable for inclusion in canonical bytes.

    Schema is fingerprinted (``schema_hash``) rather than included
    verbatim — column names can be sensitive in regulated domains
    (medical, financial). The hash is computed over the JCS-canonicalized
    parsed schema so the same logical schema produces the same hash
    regardless of MLflow's whitespace or key-ordering choices.

    Identifier fields (``name``, ``source``, ``source_type``, ``digest``,
    ``context``) are plaintext — they're the auditor-readable
    identification of the dataset and don't carry the same privacy risk
    as the column-level schema.

    Returns an empty list when the run has no logged inputs. Sorted by
    ``(name, source, context, digest)`` for determinism even when
    multiple inputs share a name.
    """
    inputs = getattr(getattr(run_data, "inputs", None), "dataset_inputs", None) or []
    out = []
    for di in inputs:
        ds = di.dataset
        context = next(
            (t.value for t in (di.tags or []) if t.key == "mlflow.data.context"),
            None,
        )
        schema_str = getattr(ds, "schema", None) or ""
        if schema_str:
            try:
                schema_canonical = canonical_json(json.loads(schema_str))
            except (ValueError, TypeError):
                # Schema isn't valid JSON — fall back to hashing the
                # raw bytes so a malformed schema can't break anchor().
                # Surface in tests if this ever fires.
                schema_canonical = schema_str.encode("utf-8")
            schema_hash = hash_data(schema_canonical)
        else:
            schema_hash = ""
        out.append({
            "name":        ds.name,
            "source":      ds.source,
            "source_type": ds.source_type,
            "digest":      ds.digest,
            "schema_hash": schema_hash,
            "context":     context,
        })
    out.sort(key=lambda d: (
        d.get("name") or "",
        d.get("source") or "",
        d.get("context") or "",
        d.get("digest") or "",
    ))
    return out


def _build_training_payload(
    *,
    run_id: str,
    params: dict,
    metrics: dict,
    artifact_checksums_map: dict,
    source_name: str,
    git_commit: str,
    mlflow_trace_id: str | None,
    metadata: dict | None,
    include_tracking_uri: bool,
    tracking_uri: str | None,
    dataset_inputs: list[dict],
) -> dict:
    """Assemble the canonical-payload dict for a training proof.

    Sorted-key serialization happens in ``canonical_json``; this function
    just decides which fields are committed to. ``metadata`` is merged
    last so callers cannot accidentally overwrite the structural fields
    above.
    """
    payload: dict = {
        "event_type": "training_complete",
        "run_id": run_id,
        "params": params,
        "metrics": metrics,
        "artifact_checksums": artifact_checksums_map,
        "source_name": source_name,
        "git_commit": git_commit,
        "dataset_inputs": dataset_inputs,
    }
    if mlflow_trace_id:
        payload["mlflow_trace_id"] = mlflow_trace_id
    if include_tracking_uri and tracking_uri:
        payload["mlflow_tracking_uri"] = tracking_uri
    if metadata:
        # Caller metadata (e.g. service_name, otel_trace_id) merges in.
        # Structural keys above win on collision so the canonical shape
        # is predictable across callers.
        for k, v in metadata.items():
            if k in payload:
                logger.debug(
                    f"Caller metadata key {k!r} collides with a structural "
                    f"field; keeping the structural value."
                )
                continue
            payload[k] = v
    return payload


def _anchor_dataset_event(
    dataset,
    *,
    proof_engine: ProofEngine,
    arweave: ArweaveAnchor,
) -> dict:
    """Anchor a standalone dataset event.

    Internal helper for ``anchor(dataset=...)``. Builds the canonical
    payload from the dataset's identity fields, signs the envelope,
    and uploads to Arweave (best-effort, same "signed-only on upload
    failure" semantics as the training-mode path).

    The dataset event commits to:

    - ``name``, ``source``, ``source_type``, ``digest`` — auditor-readable
      identifiers (plaintext).
    - ``schema_hash`` — SHA-256 of the JCS-canonicalized schema JSON.
      Column names are NOT in the proof for privacy; the hash is
      sufficient for tamper-detection.

    Notably absent: the per-run ``context`` tag (training / validation /
    test). Context is a relationship between a dataset and a specific
    *use* of it (set by ``mlflow.log_input(ds, context=...)``); it
    belongs on the per-run dataset_input reference, not on the dataset
    itself. The training event's payload still carries context per
    referenced input.

    Returns the same dict shape ``anchor()``'s training mode returns
    (envelope, payload, payload_bytes, payload_hash, anchor_result).
    Caller can read ``envelope["payload_hash"]`` or
    ``anchor_result["tx_id"]`` to chain or reference.
    """
    schema_str = getattr(dataset, "schema", None) or ""
    if schema_str:
        try:
            schema_canonical = canonical_json(json.loads(schema_str))
        except (ValueError, TypeError):
            schema_canonical = schema_str.encode("utf-8")
        schema_hash = hash_data(schema_canonical)
    else:
        schema_hash = ""

    payload: dict = {
        "event_type": "dataset",
        "name": dataset.name,
        "source": dataset.source,
        "source_type": dataset.source_type,
        "digest": dataset.digest,
        "schema_hash": schema_hash,
    }
    payload_bytes = canonical_json(payload)
    payload_hash = hash_data(payload_bytes)

    envelope = proof_engine.create_commitment(
        event_type="dataset",
        subject={
            "type": "mlflow_dataset",
            "name": dataset.name,
            "digest": dataset.digest,
        },
        payload_bytes=payload_bytes,
        # Dataset events don't have a chain-head concept yet — every
        # dataset is its own GENESIS. Dataset versioning chain
        # (`previous_hash` linking older versions of the same dataset)
        # is a deferred follow-up; see ROADMAP.
        previous_hash="GENESIS",
    )

    # Upload best-effort. Same signed-only-on-failure pattern as
    # training-mode anchor() so a transient gateway error doesn't
    # abort the caller's workflow.
    anchor_result = None
    if arweave is not None and getattr(arweave, "enabled", False):
        try:
            anchor_result = arweave.upload_proof(envelope)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Dataset upload raised for {dataset.name!r}; "
                f"keeping signed-only proof: {e}"
            )
            anchor_result = None

    return {
        "envelope": envelope,
        "payload": payload,
        "payload_bytes": payload_bytes,
        "payload_hash": payload_hash,
        "previous_hash": "GENESIS",
        "anchor_result": anchor_result,
    }


def anchor(
    proof_engine: ProofEngine | None = None,
    arweave: ArweaveAnchor | None = None,
    artifact_path: str | None = None,
    metadata: dict | None = None,
    capture_otel: bool = True,
    allow_empty_dataset_inputs: bool = False,
    *,
    dataset=None,
) -> dict:
    """Create a verifiable commitment for the current training run.

    Must be called inside an active ``mlflow.start_run()`` block, after
    artifacts have been logged. Builds a canonical payload from the
    run's params, metrics, artifact checksums, source metadata, and
    caller metadata; writes the canonical bytes as the
    ``ario/payload.json`` MLflow artifact; signs a small commitment
    envelope (~500 bytes) over the payload's SHA-256; uploads the
    envelope to Arweave; chains the proof via the registered model's
    ``ario.last_training_hash`` tag.

    See plan Part 3 (design principles) and Part 4 (plugin changes) for
    the full design rationale.

    Args:
        proof_engine: Optional override for the signing engine.
        arweave: Optional override for the Arweave anchor client.
        artifact_path: MLflow artifact subdirectory to hash. If ``None``,
            auto-resolved from MLflow's ``mlflow.log-model.history`` tag.
        metadata: Optional dict of additional fields to commit to.
            Examples: ``{"service_name": "...", "otel_trace_id": "...",
            "otel_span_id": "..."}`` for OpenTelemetry correlation,
            ``{"include_tracking_uri": True}`` to opt the
            ``mlflow_tracking_uri`` into the payload (off by default to
            avoid leaking internal infra in proofs). Caller fields are
            merged into the canonical payload after the structural fields
            (event_type, run_id, params, metrics, etc.) so structural
            fields cannot be overwritten.

    Returns:
        Dict with keys:

        - ``envelope`` — the signed pure-commitment envelope (what's on
          Arweave)
        - ``payload`` — the canonical-payload dict (what's hashed)
        - ``payload_bytes`` — the JCS-canonicalized bytes of ``payload``
        - ``payload_hash`` — SHA-256 hex of ``payload_bytes`` (also in
          ``envelope["payload_hash"]``)
        - ``previous_hash`` — chain link used (the prior training's
          ``payload_hash`` for this registered model, or ``"GENESIS"``)
        - ``registered_model`` — name of the registered model whose
          chain head was updated, or ``None`` if no registered model
          points to this run yet
        - ``anchor_result`` — Turbo upload result (``None`` if disabled
          or failed)
        - ``tags`` — the MLflow tags written on the run
        - ``artifact_path`` — the path actually used for hashing
        - ``artifact_status`` — ``"hashed"`` / ``"no_artifacts"`` /
          ``"hash_failed"``
        - ``artifact_error`` — error message when
          ``artifact_status == "hash_failed"``, else ``None``
    """
    if proof_engine is None:
        proof_engine = ProofEngine()
    if arweave is None:
        arweave = ArweaveAnchor(
            os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", ""),
            os.environ.get("ARIO_MLFLOW_GATEWAY_HOST", "turbo-gateway.com"),
        )

    # Dataset mode: dispatch to the standalone-dataset helper. No
    # active MLflow run required; caller is anchoring a Dataset object
    # standalone (publisher pattern, pre-train anchoring, etc.).
    if dataset is not None:
        return _anchor_dataset_event(
            dataset, proof_engine=proof_engine, arweave=arweave,
        )

    # Training mode (default): must be inside an active run.
    active = mlflow.active_run()
    if active is None:
        raise RuntimeError("anchor() must be called inside an active MLflow run")

    run_id = active.info.run_id
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id)

    # Auto-resolve the artifact path from the run's logged-model history when
    # the caller did not specify one. This replaces the old hardcoded default
    # of "model", which silently minted proofs with no artifact hash when the
    # caller logged under a different name.
    resolved_path = artifact_path
    if resolved_path is None:
        logged_paths = list(dict.fromkeys(_logged_model_paths(run_data)))
        if len(logged_paths) == 1:
            resolved_path = logged_paths[0]
        elif len(logged_paths) > 1:
            raise ValueError(
                f"Run {run_id} logged multiple model artifact paths "
                f"({logged_paths}); pass artifact_path explicitly so the "
                f"anchored hash matches the intended model."
            )
        else:
            resolved_path = "model"

    # Hash the model artifacts. Round metrics for cross-rerun stability —
    # JCS itself preserves exact float values, so the rounding is the
    # caller's choice and we apply it explicitly here (matches v1
    # behaviour).
    params = dict(run_data.data.params)
    metrics = normalize_floats(dict(run_data.data.metrics), precision=6)
    artifact_status = "no_artifacts"
    artifact_error: str | None = None
    try:
        checksums = artifact_checksums(run_id, artifact_path=resolved_path)
        artifact_status = "hashed" if checksums else "no_artifacts"
    except ArtifactAccessError as e:
        logger.warning(f"Skipping artifact_hash in proof for run {run_id}: {e}")
        checksums = {}
        artifact_status = "hash_failed"
        artifact_error = str(e)
    art_hash = hash_data(canonical_json(checksums)) if checksums else None

    # MLflow trace correlation when a trace is active around the training
    # call. Uncommon for training but free when present. OTel correlation
    # is the caller's job via metadata={"otel_trace_id": ...}.
    try:
        mlflow_trace_id = mlflow.get_active_trace_id()
    except Exception:  # noqa: BLE001
        mlflow_trace_id = None

    include_tracking_uri = bool(
        metadata and metadata.get("include_tracking_uri", False)
    )
    tracking_uri = mlflow.get_tracking_uri()

    # Auto-capture OTel context when default-on (capture_otel=True) and a
    # recording span is active. Caller-supplied metadata={"otel_trace_id":
    # ...} wins over auto-capture (handled in _build_training_payload's
    # merge order).
    auto_otel = capture_otel_context() if capture_otel else {}

    # Caller metadata merges with auto-OTel; caller wins on collision.
    merged_metadata = {**auto_otel, **{
        k: v for k, v in (metadata or {}).items() if k != "include_tracking_uri"
    }}

    # Read MLflow's dataset_inputs (set by mlflow.log_input). Strict-
    # by-default — anchor() refuses to mint a training proof with no
    # dataset reference, since that breaks the verifiable chain at the
    # head. allow_empty_dataset_inputs=True is the documented escape
    # hatch for the rare legitimate case (research, GPAI workflows with
    # no single dataset).
    dataset_inputs = _serialize_dataset_inputs(run_data)
    if not dataset_inputs and not allow_empty_dataset_inputs:
        raise ValueError(
            "anchor(): training run has no logged dataset inputs. "
            "Call mlflow.log_input(dataset, context=...) before training, "
            "or pass allow_empty_dataset_inputs=True to override (only "
            "do this for workflows that genuinely have no single dataset; "
            "see README on input-side anchoring)."
        )

    # Auto-anchor each dataset_input as a standalone dataset event
    # (one signed envelope per dataset, its own Arweave TX). The TX
    # is written to a run-level tag for navigation —
    # ario.dataset_anchor_tx.<dataset_name> — so the demo and any
    # chain-walking auditor can find the dataset proof from the run.
    # The TX is NOT part of the training event's canonical bytes;
    # chain integrity is provided by the inlined digest +
    # schema_hash that _serialize_dataset_inputs already includes.
    raw_inputs = list(getattr(run_data.inputs, "dataset_inputs", None) or [])
    dataset_anchors: list[dict] = []
    for di in raw_inputs:
        ds = di.dataset
        try:
            ds_result = _anchor_dataset_event(
                ds, proof_engine=proof_engine, arweave=arweave,
            )
        except Exception as e:  # noqa: BLE001
            # Don't abort training-mode anchor() on dataset-anchor failure.
            # Log and continue; the training proof still ships with
            # inlined dataset metadata (cryptographic chain integrity
            # is preserved). Best-effort matches the rest of the
            # plugin's "signed-only on transient failure" pattern.
            logger.warning(
                f"Auto-anchor of dataset {ds.name!r} raised; continuing "
                f"with training proof. Inlined metadata still ensures "
                f"chain integrity: {e}"
            )
            continue
        # Annotate the result with the dataset's name so callers can
        # match dataset_anchors entries back to dataset_inputs by name.
        ds_result["dataset_name"] = ds.name
        dataset_anchors.append(ds_result)
        ds_tx = (ds_result.get("anchor_result") or {}).get("tx_id")
        if ds_tx:
            try:
                client.set_tag(
                    run_id, f"ario.dataset_anchor_tx.{ds.name}", ds_tx,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"Could not set ario.dataset_anchor_tx.{ds.name} "
                    f"on run {run_id}: {e}"
                )

    # Build canonical payload (what's hashed and committed to).
    payload = _build_training_payload(
        run_id=run_id,
        params=params,
        metrics=metrics,
        artifact_checksums_map=checksums,
        source_name=run_data.data.tags.get("mlflow.source.name", ""),
        git_commit=run_data.data.tags.get("mlflow.source.git.commit", ""),
        mlflow_trace_id=mlflow_trace_id,
        metadata=merged_metadata,
        include_tracking_uri=include_tracking_uri,
        tracking_uri=tracking_uri,
        dataset_inputs=dataset_inputs,
    )
    payload_bytes = canonical_json(payload)
    payload_hash = hash_data(payload_bytes)

    # Find the registered model (if any) that points at this run, so we
    # can chain to its previous training proof and update the chain
    # head after a successful anchor.
    registered_model = _find_registered_model_for_run(client, run_id)
    previous_hash = "GENESIS"
    if registered_model:
        try:
            rm = client.get_registered_model(registered_model)
            existing = rm.tags.get(TAG_LAST_TRAINING_HASH) if rm and rm.tags else None
            if existing:
                previous_hash = existing
        except Exception as e:  # noqa: BLE001
            logger.debug(
                f"Could not read {TAG_LAST_TRAINING_HASH} on registered "
                f"model {registered_model!r}; using GENESIS: {e}"
            )

    # Build subject — opt-in tracking_uri respects the privacy default.
    subject: dict = {"type": "mlflow_run", "run_id": run_id}
    if include_tracking_uri:
        subject["tracking_uri"] = tracking_uri

    envelope = proof_engine.create_commitment(
        event_type="training_complete",
        subject=subject,
        payload_bytes=payload_bytes,
        previous_hash=previous_hash,
    )

    # Wrap upload_proof so a transient Turbo/Arweave outage degrades to
    # a "signed-only" outcome (anchor_result=None) rather than aborting
    # the whole anchor() call. Tags + artifacts must still be written so
    # the run carries a valid signed proof even when the upload failed.
    if arweave.enabled:
        try:
            anchor_result = arweave.upload_proof(envelope)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Arweave upload raised for run {run_id}; keeping signed-only "
                f"proof: {e}"
            )
            anchor_result = None
    else:
        anchor_result = None

    # Tags written on the run. ario.artifact_hash keeps its v1 semantics
    # (hash of artifact_checksums dict only) so VerifiedModel's load-time
    # integrity check is unchanged. ario.payload_hash is the new full
    # commitment hash.
    tags: dict = {
        "ario.public_key": envelope["public_key"],
        "ario.verify_status": "anchored" if anchor_result else "signed",
        "ario.payload_hash": payload_hash,
    }
    if art_hash is not None:
        tags["ario.artifact_hash"] = art_hash
    if anchor_result:
        tags["ario.training_tx"] = anchor_result["tx_id"]
        tags["ario.arweave_url"] = anchor_result["url"]
    wallet_mode = getattr(arweave, "wallet_mode", None)
    if wallet_mode:
        tags["ario.wallet_mode"] = wallet_mode
    wallet_type = getattr(arweave, "wallet_type", None)
    if wallet_type:
        tags["ario.wallet_type"] = wallet_type

    for key, value in tags.items():
        client.set_tag(run_id, key, value)

    # Chain head update: only after the upload succeeded so an upload
    # failure doesn't poison the head pointer with a payload that isn't
    # on Arweave. Skipped silently if no registered model exists yet —
    # the next registration via ArioMlflowClient picks up the chain
    # then.
    if registered_model and anchor_result:
        try:
            client.set_registered_model_tag(
                registered_model, TAG_LAST_TRAINING_HASH, payload_hash,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Could not write {TAG_LAST_TRAINING_HASH} on registered "
                f"model {registered_model!r} after anchor; chain may fork "
                f"on the next training of this model: {e}"
            )

    # Write artifacts: the canonical bytes (the source of truth for
    # check 2 of the verification flow), the envelope (matches what's on
    # Arweave), the Turbo receipt, and the human-readable HTML report.
    with tempfile.TemporaryDirectory() as tmpdir:
        ario_dir = os.path.join(tmpdir, "ario")
        os.makedirs(ario_dir)

        # payload.json — the canonical bytes that were committed to.
        # This is the AgentSystems-style witness: a verifier with this
        # file plus the Arweave envelope can re-hash and confirm intact.
        with open(os.path.join(ario_dir, "payload.json"), "wb") as f:
            f.write(payload_bytes)

        # proof.json — the signed envelope (matches what's on Arweave
        # exactly, byte-for-byte after JCS canonicalization).
        with open(os.path.join(ario_dir, "proof.json"), "w") as f:
            json.dump(envelope, f, indent=2)

        if anchor_result and anchor_result.get("receipt"):
            with open(os.path.join(ario_dir, "receipt.json"), "w") as f:
                json.dump(anchor_result["receipt"], f, indent=2)

        # verification.html — best-effort. The legacy report renderer
        # expects the v1 envelope shape; if it raises, log and continue
        # rather than failing the whole anchor. Phase 3 polishes this.
        try:
            html_content = generate_verification_html(
                envelope, anchor_result,
                artifact_hash=art_hash,
                wallet_mode=wallet_mode,
                wallet_type=wallet_type,
            )
            with open(os.path.join(ario_dir, "verification.html"), "w") as f:
                f.write(html_content)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"verification.html generation failed (non-fatal); legacy "
                f"renderer is incompatible with the new envelope shape and "
                f"will be updated in a later phase: {e}"
            )

        mlflow.log_artifacts(ario_dir, "ario")

    logger.info(
        f"Run {run_id} anchor complete: status={tags['ario.verify_status']}, "
        f"artifacts={artifact_status} (path={resolved_path!r}), "
        f"chain={previous_hash!r}->{payload_hash!r}, "
        f"registered_model={registered_model!r}"
    )

    return {
        "envelope": envelope,
        "payload": payload,
        "payload_bytes": payload_bytes,
        "payload_hash": payload_hash,
        "previous_hash": previous_hash,
        "registered_model": registered_model,
        "anchor_result": anchor_result,
        "tags": tags,
        "artifact_path": resolved_path,
        "artifact_status": artifact_status,
        "artifact_error": artifact_error,
        # One entry per auto-anchored dataset_input, in the order the
        # caller logged them. Each entry is the dict returned by
        # _anchor_dataset_event, plus a ``dataset_name`` key so callers
        # can match against dataset_inputs (the inlined-in-training
        # version) by name.
        "dataset_anchors": dataset_anchors,
    }
