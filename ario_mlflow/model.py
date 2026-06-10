"""VerifiedModel — inference wrapper with integrity checking and proof anchoring."""

import json
import logging
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.arweave import ArweaveAnchor
from ario_mlflow.anchoring import artifact_checksums, parse_runs_uri, capture_otel_context
from ario_mlflow.errors import (
    AssetVerificationError,
    VerifyStatusError,
    exception_for_status,
)

if TYPE_CHECKING:
    from ario_mlflow.verify_status_client import VerifyStatus, VerifyStatusClient

logger = logging.getLogger(__name__)

#: Valid values for VerifiedModel's on_failure gate policy.
ON_FAILURE_MODES = ("raise", "fail_closed", "fail_open")


class IntegrityError(AssetVerificationError):
    """Raised when model artifacts fail integrity verification.

    Subclasses :class:`~ario_mlflow.errors.AssetVerificationError` so one
    ``except AssetVerificationError`` clause catches both load-time gates
    (artifact re-hash and the agent verify-status check).
    """


def _active_trace_id() -> str | None:
    """Return the current MLflow trace id, across MLflow 2.x and 3.x.

    MLflow renamed the in-span trace-id accessor between majors:

    - **3.x** exposes ``mlflow.get_active_trace_id()`` (absent on 2.x).
    - **2.x** has no such top-level helper *and* its top-level
      ``get_last_active_trace_id()`` only populates *after* the span
      closes — both return nothing while we're still inside the
      ``@mlflow.trace`` ``predict`` span. The id is reachable only via the
      active span object, where 2.x carries it as ``request_id``
      (``trace_id`` is ``None`` on 2.x).

    Without this shim, prediction proofs on MLflow 2.x omitted
    ``mlflow_trace_id`` and never wrote the ``ario.payload_json`` trace
    tag, so prediction source-of-truth verification silently failed on
    2.x. Returns ``None`` only when tracing genuinely isn't active.
    """
    get_active = getattr(mlflow, "get_active_trace_id", None)
    if get_active is not None:
        try:
            tid = get_active()
            if tid:
                return tid
        except Exception:  # noqa: BLE001
            pass
    try:
        span = mlflow.get_current_active_span()
    except Exception:  # noqa: BLE001
        span = None
    if span is not None:
        return getattr(span, "trace_id", None) or getattr(span, "request_id", None)
    return None


def _resolve_logged_model(client, model_id: str):
    """Resolve ``models:/<model_id>`` (v3-native LoggedModel) to a duck-typed
    ModelVersion the rest of ``VerifiedModel`` can consume.

    MLflow 3 makes models first-class ``LoggedModel`` entities — `log_model()`
    returns a ``ModelInfo`` whose ``model_id`` is the canonical id, and
    ``models:/<model_id>`` is the v3-native load URI. There's no registered
    ModelVersion behind it (only a LoggedModel), so we synthesize the fields
    ``VerifiedModel`` reads:

    - ``name`` / ``version`` — the LoggedModel name + id (model_id stands in
      for version since LoggedModel is its own identity).
    - ``run_id`` / ``source`` — the source run id and the ``runs:/<rid>/<name>``
      URI, which loads fine on v3 and feeds the integrity-check artifact path.
    - ``tags = {}`` — there are no model-version registry tags here, so
      predictions chain at ``GENESIS`` (correct: there's no registration to
      chain to).

    On MLflow 2.x ``client.get_logged_model`` doesn't exist; returns ``None``
    silently so the caller falls through to the "not a v3 LoggedModel" path.
    """
    get_logged_model = getattr(client, "get_logged_model", None)
    if get_logged_model is None:
        return None
    try:
        lm = get_logged_model(model_id)
    except Exception as e:  # noqa: BLE001 — not a valid model_id (or non-v3): caller treats as "couldn't resolve"
        logger.debug(f"models:/{model_id}: not a LoggedModel id: {e}")
        return None
    source_run_id = getattr(lm, "source_run_id", None)
    name = getattr(lm, "name", None)
    if not source_run_id or not name:
        return None
    return SimpleNamespace(
        name=name,
        version=getattr(lm, "model_id", model_id),
        run_id=source_run_id,
        source=f"runs:/{source_run_id}/{name}",
        tags={},
    )


def _resolve_stage_uri_v3_fallback(client, name: str, stage: str):
    """Python-side fallback for ``models:/<name>/<stage>`` when v3 rejected
    the native search.

    MLflow 3 removed ``current_stage`` from
    ``search_model_versions``'s valid attributes, so the v2 filter string
    raises ``MlflowException: Invalid attribute key 'current_stage'``.
    Stage transitions themselves still work (deprecated-but-functional) and
    each ``ModelVersion`` still exposes ``current_stage`` — so we fetch all
    versions of the registered model and filter in Python. Returns the most
    recent version in the requested stage (matching v2 semantics).

    Aliases are the v3-native idiom; this fallback is the bridge for v2
    codebases that haven't migrated yet.
    """
    try:
        results = client.search_model_versions(f"name='{name}'")
    except Exception as e:  # noqa: BLE001 — even the fallback failed; surface None and the caller logs
        logger.warning(
            f"models:/{name}/{stage}: Python-side stage fallback failed: {e}"
        )
        return None
    in_stage = [m for m in results if getattr(m, "current_stage", None) == stage]
    if not in_stage:
        return None
    # search_model_versions returns latest-first, but the explicit sort
    # makes the "most recent version in stage" invariant defensive against
    # any future API ordering change.
    in_stage.sort(key=lambda m: int(getattr(m, "version", 0)), reverse=True)
    return in_stage[0]


def _resolve_model_version(client, model_uri: str):
    """Resolve a ``models:/`` URI to a ``ModelVersion`` using the correct MLflow API.

    Supports:

    - ``models:/<name>/<version>`` — numeric version (both majors).
    - ``models:/<name>@<alias>`` — registry alias (both majors; v3-native idiom).
    - ``models:/<name>/<stage>`` — legacy stage URI. On v2, MLflow's
      ``search_model_versions`` accepts ``current_stage`` directly. On v3 that
      attribute is gone from the search grammar, so we filter all versions of
      the registered model in Python. Stages are deprecated in 3.x — prefer
      aliases for new code.
    - ``models:/<model_id>`` — v3-native LoggedModel direct (no registered
      version behind it; returns a duck-typed handle so integrity checking
      still runs).

    Returns the resolved ``ModelVersion``-shaped object or ``None`` if the URI
    cannot be parsed or the registry lookup fails. ``None`` means
    ``VerifiedModel`` will skip the integrity check and load the URI as-is.
    """
    if not model_uri.startswith("models:/"):
        return None
    rest = model_uri[len("models:/"):]
    if not rest:
        return None

    if "@" in rest:
        name, alias = rest.split("@", 1)
        if not name or not alias:
            return None
        try:
            return client.get_model_version_by_alias(name, alias)
        except Exception as e:  # noqa: BLE001 — any MLflow-side failure (network, missing alias, perms) → None signals "couldn't resolve"
            logger.warning(f"Could not resolve alias {model_uri}: {e}")
            return None

    parts = rest.split("/", 1)
    name = parts[0]
    suffix = parts[1] if len(parts) > 1 else ""
    if not name:
        return None

    # Single-segment URI: must be a v3 LoggedModel id (``models:/<model_id>``).
    # MLflow doesn't have any other single-segment ``models:/`` form, so this
    # is the right place to try ``get_logged_model``. On v2 the helper returns
    # ``None`` and we fall through to the "couldn't resolve" path.
    if not suffix:
        return _resolve_logged_model(client, name)

    if suffix.isdigit():
        try:
            return client.get_model_version(name, suffix)
        except Exception as e:  # noqa: BLE001 — version-by-number resolution failures → None signals "couldn't resolve"
            logger.warning(f"Could not resolve version {model_uri}: {e}")
            return None

    # Stage URI (deprecated in MLflow 2.9+; the search-grammar accepts
    # ``current_stage`` on v2 but not on v3 — fall back to a Python-side
    # filter there).
    try:
        results = client.search_model_versions(
            f"name='{name}' and current_stage='{suffix}'"
        )
    except Exception as e:  # noqa: BLE001 — v3 rejects current_stage in the filter; try the Python-side fallback before giving up
        logger.debug(
            f"models:/{name}/{suffix}: native stage search failed "
            f"({e}); trying Python-side fallback"
        )
        return _resolve_stage_uri_v3_fallback(client, name, suffix)
    if not results:
        return None
    # MLflow returns latest-first; take the most recent version in the stage.
    return results[0]


@dataclass
class VerifiedPrediction:
    """Result of a verified prediction, including background anchoring status.

    Fields:
        prediction: The model's output (whatever ``pyfunc.predict`` returned).
        decision_id: UUID4 string uniquely identifying this prediction. Mirrors
            the ``ario.decision_id`` trace tag written on the MLflow trace.
        proof_status: One of:
            - ``"disabled"`` — anchoring is off (no wallet / no Turbo client).
            - ``"anchoring"`` — background upload in progress.
            - ``"anchored"`` — uploaded successfully; ``tx_id`` is set.
            - ``"failed"`` — upload raised; ``anchor_error`` is set.
        record: The canonical decision record that was signed. ``None`` only
            in exotic failure cases.
        tx_id: Arweave transaction ID, populated after a successful anchor.
        anchor_error: Stringified exception from the background anchor when
            ``proof_status == "failed"``. ``None`` otherwise.

    Use :meth:`wait_for_anchor` to block until the background thread
    finishes. The underlying :class:`threading.Event` is hidden from
    ``repr()`` and equality so it behaves like plain data otherwise.
    """
    prediction: Any
    decision_id: str
    proof_status: str  # "anchoring" | "anchored" | "disabled" | "failed"
    record: dict | None = None
    tx_id: str | None = None
    anchor_error: str | None = None
    _anchor_done: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )

    def wait_for_anchor(self, timeout: float | None = None) -> bool:
        """Block until the background anchor completes or the timeout expires.

        Args:
            timeout: Maximum seconds to wait. ``None`` waits forever.

        Returns:
            ``True`` if the background anchor finished (check ``proof_status``,
            ``tx_id``, and ``anchor_error`` for outcome). ``False`` if the
            timeout expired while still ``"anchoring"``.

        When anchoring is disabled (``proof_status == "disabled"``) the event
        is already set and this returns ``True`` immediately.
        """
        return self._anchor_done.wait(timeout=timeout)


class VerifiedModel:
    """Wraps an MLflow model with integrity checking and proof anchoring on predict()."""

    def __init__(
        self,
        model_uri: str,
        proof_engine: ProofEngine | None = None,
        anchor: ArweaveAnchor | None = None,
        *,
        asset_id: str | None = None,
        verify_status_client: "VerifyStatusClient | None" = None,
        on_failure: str = "raise",
    ):
        """Load an MLflow model and verify its artifacts against the anchored hash.

        Resolves ``model_uri`` through the MLflow registry, re-hashes the
        model artifacts, and compares the result to the ``ario.artifact_hash``
        tag from the source training run. The integrity check runs **before**
        :func:`mlflow.pyfunc.load_model`, so a tampered artifact is rejected
        before any user code (``PythonModel`` subclasses, custom loaders) can
        execute.

        When ``asset_id`` + ``verify_status_client`` are provided, an
        **agent verify-status gate** additionally runs first — before the
        artifact integrity check and before any MLflow access (one cheap
        local HTTP call; fail fast before paying for artifact downloads).
        It consults ar-io-agent's ``GET /v1/verify-status/<asset_id>`` and
        applies the ``on_failure`` policy to the §9.1 outcome mapping:
        ``tampered`` → :class:`~ario_mlflow.errors.AssetTamperedError`,
        ``missing`` → :class:`~ario_mlflow.errors.AssetMissingError`,
        ``unavailable`` or ``verified``-but-stale →
        :class:`~ario_mlflow.errors.AssetStaleError`, ``unknown`` →
        :class:`~ario_mlflow.errors.AssetUnknownError`. Transport-level
        failures (:class:`~ario_mlflow.errors.VerifyStatusTransportError`
        etc.) fall under the same policy. The gate runs at **load time
        only** — a tamper detected by the agent *after* this constructor
        returns is not re-checked on ``predict()`` (per-predict re-checking
        with the contract's 10–30s cache guidance is a planned follow-up).

        Args:
            model_uri: A ``models:/`` URI in any of these forms:

                - ``models:/<name>/<version>`` — numeric version.
                - ``models:/<name>@<alias>`` — registry alias.
                - ``models:/<name>/<stage>`` — legacy stage URI (MLflow's
                  ``search_model_versions`` is used; deprecated in 2.9+).
            proof_engine: Override for the signing engine. Defaults to a
                :class:`ProofEngine` using the process-local Ed25519 key.
            anchor: Override for the Arweave anchor client. Defaults to an
                :class:`ArweaveAnchor` configured from the
                ``ARIO_MLFLOW_ARWEAVE_WALLET`` /
                ``ARIO_MLFLOW_GATEWAY_HOST`` env vars.
            asset_id: The ar-io-agent policy asset_id covering this model's
                artifacts. Requires ``verify_status_client``.
            verify_status_client: A
                :class:`~ario_mlflow.verify_status_client.VerifyStatusClient`
                pointed at the agent's management port (same host) or the
                api-guard proxy. Requires ``asset_id``.
            on_failure: Gate policy — ``"raise"`` (default) and
                ``"fail_closed"`` raise the typed exception (identical
                behavior; the latter name reads better in production
                configs); ``"fail_open"`` logs at WARN with structured
                fields and proceeds with the load. Applies only to the
                verify-status gate; :class:`IntegrityError` always raises.

        Raises:
            IntegrityError: If the re-hashed artifacts do not match the
                ``ario.artifact_hash`` anchored at training time. The underlying
                pyfunc model is never loaded in this case.
            ario_mlflow.errors.VerifyStatusError: (subclasses) when the
                verify-status gate refuses the load and ``on_failure`` is
                ``"raise"``/``"fail_closed"``. The pyfunc model is never
                loaded and no MLflow call is made in this case.
            ValueError: On an invalid ``on_failure`` value, or when only one
                of ``asset_id`` / ``verify_status_client`` is provided.
        """
        if on_failure not in ON_FAILURE_MODES:
            raise ValueError(
                f"on_failure must be one of {ON_FAILURE_MODES}, got {on_failure!r}"
            )
        if (asset_id is None) != (verify_status_client is None):
            raise ValueError(
                "asset_id and verify_status_client must be provided together"
            )
        self._asset_id = asset_id
        self._verify_status_client = verify_status_client
        self._on_failure = on_failure

        # Agent verify-status gate FIRST: one cheap local HTTP read against
        # agent state. Artifact integrity (downloads + hashing) and pyfunc
        # load only happen once this passes (or fail_open logs through).
        if verify_status_client is not None:
            self._gate_on_verify_status()

        self._model_uri = model_uri
        self._proof_engine = proof_engine or ProofEngine()
        self._anchor = anchor or ArweaveAnchor(
            os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", ""),
            os.environ.get("ARIO_MLFLOW_GATEWAY_HOST", "turbo-gateway.com"),
        )

        client = mlflow.tracking.MlflowClient()
        # Reused for trace-tag writes on the predict path. The top-level
        # ``mlflow.set_trace_tag`` only exists on MLflow 3.x; the client
        # method exists on both 2.x and 3.x.
        self._mlflow_client = client
        self.model_name = "unknown"
        self.model_version = "unknown"
        self.run_id = "unknown"

        # Resolve the models:/ URI via the correct MLflow registry API for each
        # supported URI form:
        #   models:/<name>/<numeric_version>  → get_model_version
        #   models:/<name>@<alias>            → get_model_version_by_alias
        #   models:/<name>/<stage>            → search_model_versions (deprecated)
        mv = _resolve_model_version(client, model_uri)
        if mv is not None:
            self.model_name = mv.name
            self.model_version = str(mv.version)
            self.run_id = mv.run_id or "unknown"

        # ModelVersion.source preserves the original artifact path from
        # registration (e.g. "sklearn-model") — we must use it rather than
        # hardcoding "/model".
        load_uri = model_uri
        artifact_path = "model"
        if mv is not None and mv.source:
            load_uri = mv.source
            _src_run_id, src_artifact_path = parse_runs_uri(mv.source)
            if src_artifact_path:
                artifact_path = src_artifact_path

        # Verify artifact integrity BEFORE loading the model. pyfunc models can
        # execute user code during load (PythonModel subclasses, custom loaders),
        # so a tampered artifact must be rejected before mlflow.pyfunc.load_model
        # is given a chance to run it.
        self._artifact_verified = None
        if self.run_id != "unknown":
            try:
                run = client.get_run(self.run_id)
                expected_hash = run.data.tags.get("ario.artifact_hash")
                if expected_hash:
                    checksums = artifact_checksums(self.run_id, artifact_path=artifact_path)
                    if not checksums:
                        logger.warning(
                            f"Could not download artifacts for integrity check of {model_uri}; "
                            f"treating status as unknown"
                        )
                    else:
                        computed_hash = hash_data(canonical_json(checksums))
                        if computed_hash != expected_hash:
                            raise IntegrityError(
                                f"Model artifact integrity check failed for {model_uri}. "
                                f"Expected {expected_hash}, got {computed_hash}"
                            )
                        self._artifact_verified = True
                        logger.info(f"Artifact integrity verified for {model_uri}")
            except IntegrityError:
                raise
            except Exception as e:  # noqa: BLE001 — IntegrityError already re-raised above; everything else (mlflow access, file IO) is logged-and-skipped to avoid blocking model load
                logger.warning(f"Could not verify artifact integrity: {e}")

        # Integrity has passed (or was unverifiable with a logged warning).
        # Only now load the model.
        self._model = mlflow.pyfunc.load_model(load_uri)

        # Per-prediction chain head: predictions chain to ario.registration_tx
        # on the model version (read once at __init__ from the already-
        # resolved ModelVersion; no tag write at predict time — eliminates
        # the high-frequency busy-case race). See plan Part 3 design
        # principle 5 and Part 4 plugin change item 2.
        #
        # Trade-off: tags are read from the mv resolved at __init__ time.
        # If a registration's background anchor thread is still in flight
        # when VerifiedModel is constructed, the tag may not be there yet
        # and predictions for this instance chain at GENESIS. Acceptable
        # because (a) typical workflows wait for registration before
        # serving, (b) future VerifiedModel instances pick up the tag,
        # and (c) the chain is reconstructable from Arweave by tag query
        # — no proof is silently lost.
        self._prediction_previous_hash = "GENESIS"
        if mv is not None:
            mv_tags = getattr(mv, "tags", None) or {}
            reg_tx = mv_tags.get("ario.registration_tx")
            if reg_tx:
                self._prediction_previous_hash = reg_tx

        self._lock = threading.Lock()

    def _gate_on_verify_status(self) -> None:
        """Consult the agent's verify-status endpoint and apply on_failure.

        Outcome→exception mapping is `verify-status-api.md` §9.1 via
        :func:`ario_mlflow.errors.exception_for_status`. Transport-level
        client errors (auth, 404, network, the api-guard 503 license gate)
        fall under the same ``on_failure`` policy — contract §9.1: "treat
        any non-200 as a transport error and apply the same on_failure
        policy."
        """
        try:
            status = self._verify_status_client.get(self._asset_id)
        except VerifyStatusError as e:
            self._apply_gate_policy(e, e.status)
            return
        exc = exception_for_status(status)
        if exc is not None:
            self._apply_gate_policy(exc, status)
            return
        logger.info(
            f"Agent verify-status gate passed for asset {self._asset_id!r} "
            f"(outcome=verified, stale=False)"
        )

    def _apply_gate_policy(
        self, exc: VerifyStatusError, status: "VerifyStatus | None"
    ) -> None:
        """Raise per on_failure, or log-and-proceed for fail_open.

        A model load that silently bypasses verification is a regulatory
        liability — the fail_open line is WARN with structured fields
        (``extra["ario_verify_status"]``) so SIEM pipelines can key on it.
        """
        if self._on_failure != "fail_open":
            raise exc
        fields = {
            "asset_id": self._asset_id,
            "error": type(exc).__name__,
            "outcome": status.outcome if status else None,
            "stale": status.stale if status else None,
            "policy_hash": status.policy_hash if status else None,
            "current_tx_id": status.current_tx_id if status else None,
        }
        logger.warning(
            "fail_open: proceeding with model load DESPITE a failed "
            "verification gate — "
            + " ".join(f"{k}={v}" for k, v in fields.items())
            + f" detail={exc}",
            extra={"ario_verify_status": fields},
        )

    def _build_prediction_payload(
        self,
        *,
        decision_id: str,
        input_hash: str,
        output_hash: str,
        latency_ms: float,
        mlflow_trace_id: str | None,
        metadata: dict | None,
    ) -> dict:
        """Assemble the canonical payload for a prediction commitment.

        Privacy-preserving by construction: contains hashes of input/output,
        not the values themselves. PII / customer data stays in the demo
        cache (or wherever the caller persists raw values) — the proof
        only commits to fingerprints. See plan Part 3 (Arweave is a
        witness, MLflow is the system of record).
        """
        payload: dict = {
            "event_type": "prediction",
            "decision_id": decision_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "run_id": self.run_id,
            "model_uri": self._model_uri,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "latency_ms": round(latency_ms, 2),
            "artifact_verified": self._artifact_verified,
        }
        if mlflow_trace_id:
            payload["mlflow_trace_id"] = mlflow_trace_id
        if metadata:
            for k, v in metadata.items():
                if k in payload:
                    logger.debug(
                        f"Caller metadata key {k!r} collides with a structural "
                        f"field; keeping the structural value."
                    )
                    continue
                payload[k] = v
        return payload

    @mlflow.trace(name="VerifiedModel.predict")
    def predict(
        self,
        input_data,
        *,
        metadata: dict | None = None,
        capture_otel: bool = True,
    ) -> VerifiedPrediction:
        """Run inference, sign a pure-commitment proof, and anchor asynchronously.

        Args:
            input_data: A dict of named features, a list/tuple of positional
                features, or any array-like the underlying pyfunc model
                accepts. Dicts and single-row lists are wrapped into a
                2-D array (``[[values]]``) before passing to the model.
            metadata: Optional dict of additional fields to commit to in
                the canonical payload. Examples: ``{"otel_trace_id": "...",
                "otel_span_id": "...", "service_name": "..."}`` for
                OpenTelemetry correlation, or any other caller-shaped
                fields. Structural fields cannot be overwritten.

        Returns:
            A :class:`VerifiedPrediction`. ``prediction`` is whatever the
            wrapped model returned. The Arweave upload runs in a
            background thread; callers that need ``tx_id`` immediately
            should call :meth:`VerifiedPrediction.wait_for_anchor` before
            reading it.

        Side effects:
            - The ``@mlflow.trace`` span is annotated with ``ario.*`` tags
              that mirror the canonical payload (``decision_id``,
              ``model_name``, ``model_version``, ``input_hash``,
              ``output_hash``, ``payload_hash``, ``proof_status``,
              ``artifact_verified``). The mirrored tags let an MLflow-UI
              user see what was committed without downloading the
              envelope; verifiers should still re-derive canonical bytes
              from the source values rather than trusting the tags.
            - If :class:`ArweaveAnchor` is enabled, the envelope (~500
              bytes) is uploaded to Arweave in a daemon thread. Errors
              surface on the returned object, not raised.
        """
        decision_id = str(uuid.uuid4())
        start = time()

        if isinstance(input_data, dict):
            input_array = np.array([list(input_data.values())])
        elif isinstance(input_data, (list, tuple)):
            input_array = np.array([input_data])
        else:
            input_array = input_data

        prediction = self._model.predict(input_array)
        latency_ms = (time() - start) * 1000

        if hasattr(prediction, 'tolist'):
            pred_serializable = prediction.tolist()
        else:
            pred_serializable = prediction

        input_serializable = input_data if isinstance(input_data, dict) else {"features": list(input_data) if hasattr(input_data, '__iter__') else input_data}
        input_hash = hash_data(canonical_json(input_serializable))
        output_hash = hash_data(canonical_json({"prediction": pred_serializable}))

        # Capture MLflow trace_id (free correlation when @mlflow.trace
        # span is active; OTel context flows in via metadata). Resolved
        # across MLflow 2.x/3.x — see _active_trace_id.
        trace_id = _active_trace_id()

        # Auto-capture OTel context when default-on. Caller-supplied
        # metadata={"otel_trace_id": ...} wins on collision via the merge
        # order in _build_prediction_payload.
        auto_otel = capture_otel_context() if capture_otel else {}
        merged_metadata = {**auto_otel, **(metadata or {})}

        payload = self._build_prediction_payload(
            decision_id=decision_id,
            input_hash=input_hash,
            output_hash=output_hash,
            latency_ms=latency_ms,
            mlflow_trace_id=trace_id,
            metadata=merged_metadata,
        )
        payload_bytes = canonical_json(payload)
        payload_hash = hash_data(payload_bytes)

        # Subject identifies WHERE the verifier looks up the source data.
        # For predictions, the canonical bytes live as an MLflow artifact
        # on the model's source run at ario/predictions/<decision_id>/payload.json.
        # The verifier needs (run_id, decision_id) to find that artifact;
        # trace_id is included for observability correlation when present.
        subject: dict = {
            "type": "mlflow_prediction",
            "decision_id": decision_id,
            "model_run_id": self.run_id,
        }
        if trace_id:
            subject["trace_id"] = trace_id

        # All predictions for this model chain to the registration that
        # produced it (read once at __init__). No tag write at predict
        # time — predictions for the same model fork into a tree (one
        # per call) which is the natural shape; the chain audit walks
        # via Arweave tag query, not via a shared head.
        with self._lock:
            envelope = self._proof_engine.create_commitment(
                event_type="prediction",
                subject=subject,
                payload_bytes=payload_bytes,
                previous_hash=self._prediction_previous_hash,
            )

        # Write the canonical bytes as an MLflow artifact on the model's
        # source run. This is the per-prediction equivalent of training's
        # ario/payload.json — it gives the verifier an immutable witness
        # to download for check 2 (anchored bytes intact). Trace tags
        # (set further down) are for observability/UI display only and
        # MUST NOT be treated as authoritative; the artifact is the
        # source of truth.
        if self.run_id and self.run_id != "unknown":
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    pred_dir = os.path.join(tmpdir, "predictions", decision_id)
                    os.makedirs(pred_dir)
                    with open(os.path.join(pred_dir, "payload.json"), "wb") as f:
                        f.write(payload_bytes)
                    with open(os.path.join(pred_dir, "proof.json"), "w") as f:
                        json.dump(envelope, f, indent=2)
                    mlflow.log_artifacts(
                        local_dir=os.path.dirname(os.path.dirname(pred_dir)),
                        artifact_path="ario",
                        run_id=self.run_id,
                    )
            except Exception as e:  # noqa: BLE001
                # Artifact write failure is non-fatal — the proof is on
                # Arweave, the signature still verifies. Check 2 won't be
                # available for this prediction until the artifact is
                # written. Log and continue serving.
                logger.warning(
                    f"Could not write ario/predictions/{decision_id}/payload.json "
                    f"on run {self.run_id}: {e}. Prediction is still anchored to "
                    f"Arweave and signature-verifiable; check 2 will report "
                    f"payload_artifact_not_available."
                )

        result = VerifiedPrediction(
            prediction=prediction,
            decision_id=decision_id,
            proof_status="disabled" if not self._anchor.enabled else "anchoring",
            record=payload,  # the canonical payload is what we committed to
        )
        if not self._anchor.enabled:
            result._anchor_done.set()

        # Mirror the canonical payload onto the trace as ``ario.payload_json``
        # so verify_source_of_truth has an independent MLflow surface to
        # compare against the artifact (parallel to how training's
        # source-of-truth check re-fetches run.data.params/metrics).
        # The individual ario.* observability tags below are mirrors for
        # MLflow-UI users; ario.payload_json is what verify_source_of_truth
        # reads at audit time.
        if trace_id:
            try:
                self._mlflow_client.set_trace_tag(
                    trace_id, "ario.payload_json", payload_bytes.decode("utf-8"),
                )
                self._mlflow_client.set_trace_tag(trace_id, "ario.decision_id", decision_id)
                self._mlflow_client.set_trace_tag(trace_id, "ario.model_name", self.model_name)
                self._mlflow_client.set_trace_tag(trace_id, "ario.model_version", self.model_version)
                self._mlflow_client.set_trace_tag(trace_id, "ario.input_hash", input_hash)
                self._mlflow_client.set_trace_tag(trace_id, "ario.output_hash", output_hash)
                self._mlflow_client.set_trace_tag(trace_id, "ario.payload_hash", payload_hash)
                self._mlflow_client.set_trace_tag(trace_id, "ario.proof_status", result.proof_status)
                if self._artifact_verified is not None:
                    self._mlflow_client.set_trace_tag(
                        trace_id, "ario.artifact_verified",
                        str(self._artifact_verified).lower(),
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to tag MLflow trace {trace_id}: {e}")

        if self._anchor.enabled:
            threading.Thread(
                target=self._anchor_prediction,
                args=(result, envelope, trace_id),
                daemon=True,
            ).start()

        return result

    def _anchor_prediction(
        self,
        result: VerifiedPrediction,
        envelope: dict,
        trace_id: str | None = None,
    ):
        """Background: upload prediction commitment to Arweave; update trace."""
        try:
            anchor_result = self._anchor.upload_proof(envelope)
            if anchor_result:
                result.tx_id = anchor_result["tx_id"]
                result.proof_status = "anchored"
                if trace_id:
                    try:
                        self._mlflow_client.set_trace_tag(trace_id, "ario.prediction_tx", anchor_result["tx_id"])
                        self._mlflow_client.set_trace_tag(trace_id, "ario.arweave_url", anchor_result["url"])
                        self._mlflow_client.set_trace_tag(trace_id, "ario.proof_status", "anchored")
                    except Exception as e:  # noqa: BLE001
                        logger.debug(
                            f"Could not update trace {trace_id} with anchor tags: {e}"
                        )
                logger.info(
                    f"Prediction {result.decision_id} anchored: tx={anchor_result['tx_id']}"
                )
            else:
                result.proof_status = "failed"
                result.anchor_error = "upload returned no result"
                if trace_id:
                    try:
                        self._mlflow_client.set_trace_tag(trace_id, "ario.proof_status", "failed")
                    except Exception as trace_error:  # noqa: BLE001
                        logger.debug(
                            f"Could not update trace {trace_id} with failed status: {trace_error}"
                        )
                logger.error(
                    f"Prediction anchoring failed for {result.decision_id}: upload returned no result"
                )
        except Exception as e:  # noqa: BLE001
            result.proof_status = "failed"
            result.anchor_error = str(e)
            if trace_id:
                try:
                    self._mlflow_client.set_trace_tag(trace_id, "ario.proof_status", "failed")
                except Exception as trace_error:  # noqa: BLE001
                    logger.debug(
                        f"Could not update trace {trace_id} with failed status: {trace_error}"
                    )
            logger.error(f"Prediction anchoring failed for {result.decision_id}: {e}")
        finally:
            result._anchor_done.set()
