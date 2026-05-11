"""ArioMlflowClient — wraps MlflowClient with automatic proof anchoring."""

import json
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone

from mlflow.tracking import MlflowClient

from ario_mlflow.proof import ProofEngine, canonical_json, hash_data
from ario_mlflow.arweave import ArweaveAnchor
from ario_mlflow.anchoring import (
    artifact_checksums,
    parse_runs_uri,
    ArtifactAccessError,
    capture_otel_context,
)
from ario_mlflow.report import generate_verification_html

logger = logging.getLogger(__name__)


class ArioMlflowClient(MlflowClient):
    """MlflowClient that auto-anchors model registration and promotion events.

    Anchoring runs in a daemon thread so the MLflow call returns immediately.
    Because the return value (an MLflow ``ModelVersion``) has no room for an
    anchor future, the client exposes status via two methods:

    - :meth:`anchor_status` — returns the latest status for a given event.
    - :meth:`wait_for_anchor` — blocks until that event finishes.

    Statuses are keyed by ``(event_type, name, version)``, where
    ``event_type`` is ``"registration"`` or ``"promotion"``.
    """

    def __init__(self, *args, proof_engine: ProofEngine | None = None, anchor: ArweaveAnchor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._proof_engine = proof_engine or ProofEngine()
        self._anchor = anchor or ArweaveAnchor(
            os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", ""),
            os.environ.get("ARIO_MLFLOW_GATEWAY_HOST", "turbo-gateway.com"),
        )
        # Per-(event_type, name, version) status and completion events, so the
        # caller can observe async anchoring outcomes without the ModelVersion
        # return value carrying a future.
        self._anchor_events: dict[tuple[str, str, str], threading.Event] = {}
        self._anchor_statuses: dict[tuple[str, str, str], dict] = {}
        self._anchor_state_lock = threading.Lock()

    def _status_key(self, event_type: str, name: str, version: str) -> tuple[str, str, str]:
        return (event_type, name, str(version))

    def _ensure_anchor_state(self) -> None:
        """Lazily initialize the status-tracking attributes.

        Subclasses that override ``__init__`` without calling super (e.g.
        test doubles) still work — the first call that needs these
        attributes creates them.
        """
        if not hasattr(self, "_anchor_state_lock"):
            self._anchor_state_lock = threading.Lock()
            self._anchor_events = {}
            self._anchor_statuses = {}

    def _register_pending(self, event_type: str, name: str, version: str) -> threading.Event:
        self._ensure_anchor_state()
        key = self._status_key(event_type, name, version)
        event = threading.Event()
        with self._anchor_state_lock:
            self._anchor_events[key] = event
            self._anchor_statuses[key] = {
                "status": "anchoring",
                "error": None,
                "tx_id": None,
            }
        return event

    def _record_status(
        self,
        event_type: str,
        name: str,
        version: str,
        status: str,
        tx_id: str | None = None,
        error: str | None = None,
    ) -> None:
        self._ensure_anchor_state()
        key = self._status_key(event_type, name, version)
        with self._anchor_state_lock:
            self._anchor_statuses[key] = {
                "status": status,
                "error": error,
                "tx_id": tx_id,
            }

    def anchor_status(self, event_type: str, name: str, version: str) -> dict:
        """Return the latest anchor status for a registration or promotion.

        Args:
            event_type: ``"registration"`` or ``"promotion"``.
            name: Registered model name.
            version: Model version (int or string).

        Returns:
            A dict with keys ``status`` (``"anchoring"`` | ``"anchored"`` |
            ``"signed"`` | ``"failed"`` | ``"unknown"``), ``tx_id``,
            ``error``, and ``done`` (bool — has the background thread
            finished).
        """
        self._ensure_anchor_state()
        key = self._status_key(event_type, name, version)
        with self._anchor_state_lock:
            status = dict(self._anchor_statuses.get(key, {"status": "unknown", "error": None, "tx_id": None}))
            event = self._anchor_events.get(key)
        status["done"] = bool(event and event.is_set())
        return status

    def wait_for_anchor(
        self,
        event_type: str,
        name: str,
        version: str,
        timeout: float | None = None,
    ) -> bool:
        """Block until the background anchor for this event completes.

        Returns ``True`` if the anchor finished (check :meth:`anchor_status`
        for the outcome). ``False`` if the timeout expired, or if no anchor
        was ever queued for this key.
        """
        self._ensure_anchor_state()
        key = self._status_key(event_type, name, version)
        with self._anchor_state_lock:
            event = self._anchor_events.get(key)
        if event is None:
            return False
        return event.wait(timeout=timeout)

    def create_model_version(
        self,
        name,
        source,
        run_id=None,
        *,
        metadata=None,
        capture_otel: bool = True,
        **kwargs,
    ):
        """Register a model version and anchor a pure-commitment proof.

        Args:
            name: Registered model name.
            source: ``runs:/<run_id>/<artifact_path>`` URI of the source.
            run_id: Optional explicit run ID. If absent, parsed from
                ``source``.
            metadata: Optional extra fields merged into the canonical
                payload (e.g. caller metadata, OTel correlation).
                Structural fields cannot be overwritten.
            **kwargs: Forwarded to ``MlflowClient.create_model_version``.

        See plan Part 4 plugin change item 2 for the chain semantics:
        registration proofs read ``ario.training_tx`` from the source
        run and use it as ``previous_hash``. No tag write at
        registration — the chain head pointer (``ario.last_training_hash``)
        was set by ``anchor()``.
        """
        mv = super().create_model_version(name, source, run_id=run_id, **kwargs)
        event = self._register_pending("registration", name, str(mv.version))

        # Capture OTel context HERE on the calling thread — the daemon
        # thread won't have the parent's OTel context, so we snapshot
        # before dispatching.
        auto_otel = capture_otel_context() if capture_otel else {}
        merged_metadata = {**auto_otel, **(metadata or {})}

        threading.Thread(
            target=self._anchor_registration,
            args=(name, str(mv.version), run_id, source),
            kwargs={"metadata": merged_metadata, "done_event": event},
            daemon=True,
        ).start()

        return mv

    def transition_model_version_stage(
        self,
        name,
        version,
        stage,
        *,
        metadata=None,
        capture_otel: bool = True,
        **kwargs,
    ):
        """Transition a model stage and anchor a pure-commitment proof.

        Args:
            name: Registered model name.
            version: Model version.
            stage: Target stage.
            metadata: Optional extra fields merged into the canonical
                payload.
            **kwargs: Forwarded to ``MlflowClient.transition_model_version_stage``.

        Promotion proofs chain to ``ario.registration_tx`` on the model
        version.
        """
        current = self.get_model_version(name, version)
        from_stage = current.current_stage

        result = super().transition_model_version_stage(name, version, stage, **kwargs)
        event = self._register_pending("promotion", name, str(version))

        # Snapshot OTel context on the calling thread — daemon thread
        # has its own context.
        auto_otel = capture_otel_context() if capture_otel else {}
        merged_metadata = {**auto_otel, **(metadata or {})}

        threading.Thread(
            target=self._anchor_promotion,
            args=(name, str(version), from_stage, stage),
            kwargs={"metadata": merged_metadata, "done_event": event},
            daemon=True,
        ).start()

        return result

    def _build_registration_payload(
        self,
        *,
        model_name: str,
        version: str,
        source_run_id: str | None,
        source: str | None,
        artifact_verified: bool | None,
        artifact_hash: str | None,
        metadata: dict | None,
    ) -> dict:
        """Assemble the canonical payload for a registration commitment."""
        payload: dict = {
            "event_type": "model_registered",
            "model_name": model_name,
            "model_version": version,
            "source_run_id": source_run_id,
            "source": source,
            "artifact_verified": artifact_verified,
            "artifact_hash": artifact_hash,
        }
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

    def _anchor_registration(
        self,
        model_name: str,
        version: str,
        run_id: str | None,
        source: str | None,
        *,
        metadata: dict | None = None,
        done_event: threading.Event | None = None,
    ):
        """Background: verify artifact integrity, anchor a pure-commitment registration proof."""
        try:
            training_tx = None
            expected_hash = None
            artifact_verified = None

            # When run_id is absent, derive it from the source URI so the
            # registration proof still links to the training run rather
            # than minting a fresh GENESIS chain. When BOTH are present
            # and disagree, fail loudly: silently preferring run_id
            # would mint a structurally inconsistent proof — the chain
            # link would point to one run while the signed payload's
            # ``source`` field claims a different one.
            src_run_id, src_artifact_path = parse_runs_uri(source)
            if run_id and src_run_id and run_id != src_run_id:
                raise ValueError(
                    f"run_id {run_id!r} does not match source URI run_id "
                    f"{src_run_id!r}; refusing to mint a proof with "
                    f"inconsistent provenance"
                )
            source_run_id = run_id or src_run_id

            if source_run_id:
                try:
                    run = self.get_run(source_run_id)
                    training_tx = run.data.tags.get("ario.training_tx")
                    expected_hash = run.data.tags.get("ario.artifact_hash")
                except Exception as e:  # noqa: BLE001 — tracking-store read failure: skip anchor attempt rather than mint a fresh GENESIS chain (would permanently break provenance — see comment below)
                    # A transient tracking-store failure must NOT silently
                    # drop training_tx and mint a fresh GENESIS chain —
                    # that would permanently break provenance for this
                    # model version. Skip anchoring this attempt instead.
                    logger.warning(
                        f"Skipping registration anchoring for {model_name}/v{version}: "
                        f"could not load source run {source_run_id}: {e}"
                    )
                    self._record_status(
                        "registration", model_name, version,
                        status="failed",
                        error=f"Could not load source run {source_run_id}: {e}",
                    )
                    return

                try:
                    checksums = artifact_checksums(
                        source_run_id, artifact_path=src_artifact_path or "model",
                    )
                except ArtifactAccessError as e:
                    # Can't verify — leave artifact_verified as None
                    # (unknown) and continue anchoring the event itself.
                    logger.warning(
                        f"Could not re-hash artifacts for {model_name}/v{version}: {e}"
                    )
                    checksums = {}
                if checksums and expected_hash is not None:
                    computed_hash = hash_data(canonical_json(checksums))
                    artifact_verified = computed_hash == expected_hash

            payload = self._build_registration_payload(
                model_name=model_name,
                version=version,
                source_run_id=source_run_id,
                source=source,
                artifact_verified=artifact_verified,
                artifact_hash=expected_hash,
                metadata=metadata,
            )
            payload_bytes = canonical_json(payload)
            payload_hash = hash_data(payload_bytes)

            envelope = self._proof_engine.create_commitment(
                event_type="model_registered",
                subject={
                    "type": "mlflow_model_version",
                    "name": model_name,
                    "version": str(version),
                },
                payload_bytes=payload_bytes,
                previous_hash=training_tx or "GENESIS",
            )

            # Defense in depth: wrap upload_proof so a transient
            # Turbo/Arweave outage degrades to signed-only (result=None)
            # rather than aborting the whole _anchor_registration body
            # via the outer except. Tags + payload.json artifact must
            # still be written so the model version carries a valid
            # signed proof even when the upload failed. Symmetric with
            # anchor()'s upload_proof wrapping.
            if self._anchor.enabled:
                try:
                    result = self._anchor.upload_proof(envelope)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"Registration upload raised for {model_name}/v{version}; "
                        f"keeping signed-only proof: {e}"
                    )
                    result = None
            else:
                result = None

            tags = {
                "ario.verify_status": "anchored" if result else "signed",
                "ario.public_key": envelope["public_key"],
                "ario.payload_hash": payload_hash,
            }
            if artifact_verified is not None:
                tags["ario.artifact_verified"] = str(artifact_verified).lower()
            if result:
                tags["ario.registration_tx"] = result["tx_id"]
                tags["ario.arweave_url"] = result["url"]
            wallet_mode = getattr(self._anchor, "wallet_mode", None)
            if wallet_mode:
                tags["ario.wallet_mode"] = wallet_mode

            for key, value in tags.items():
                self.set_model_version_tag(model_name, version, key, value)

            with tempfile.TemporaryDirectory() as tmpdir:
                ario_dir = os.path.join(tmpdir, "ario")
                os.makedirs(ario_dir)

                # registration_payload.json — the canonical bytes (the
                # AgentSystems-style witness for check 2).
                with open(os.path.join(ario_dir, "registration_payload.json"), "wb") as f:
                    f.write(payload_bytes)

                # registration_proof.json — the signed envelope (matches
                # what was uploaded to Arweave).
                with open(os.path.join(ario_dir, "registration_proof.json"), "w") as f:
                    json.dump(envelope, f, indent=2)

                if result and result.get("receipt"):
                    with open(os.path.join(ario_dir, "registration_receipt.json"), "w") as f:
                        json.dump(result["receipt"], f, indent=2)

                # verification.html — best-effort. Legacy renderer expects
                # the v1 envelope shape; if it raises, log and continue
                # rather than failing the whole anchor. Phase 3 will
                # rebuild the report for the new shape.
                try:
                    report = generate_verification_html(
                        envelope, result,
                        artifact_hash=expected_hash,
                        artifact_verified=artifact_verified,
                        cli_verify_cmd=f"ar-io-mlflow verify model {model_name}/{version}",
                        wallet_mode=wallet_mode,
                    )
                    with open(os.path.join(ario_dir, "registration_verification.html"), "w") as f:
                        f.write(report)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"registration_verification.html generation failed (non-fatal): {e}"
                    )

                if source_run_id:
                    self.log_artifacts(source_run_id, ario_dir, "ario")

            status = "anchored" if result else "signed (anchoring disabled or upload failed)"
            logger.info(
                f"Registration {model_name}/v{version} {status}: "
                f"verified={artifact_verified}, "
                f"chain={training_tx or 'GENESIS'!r}->{payload_hash!r}"
            )

            self._record_status(
                "registration", model_name, version,
                status="anchored" if result else "signed",
                tx_id=result["tx_id"] if result else None,
            )

        except Exception as e:  # noqa: BLE001 — registration anchoring runs in a daemon thread; failure must record status="failed" and continue, not crash the calling thread
            logger.error(f"Failed to anchor registration {model_name}/v{version}: {e}")
            self._record_status(
                "registration", model_name, version,
                status="failed",
                error=str(e),
            )
        finally:
            if done_event is not None:
                done_event.set()

    def _anchor_promotion(
        self,
        model_name: str,
        version: str,
        from_stage: str,
        to_stage: str,
        *,
        metadata: dict | None = None,
        done_event: threading.Event | None = None,
    ):
        """Background: anchor a pure-commitment stage-transition proof."""
        try:
            mv = self.get_model_version(model_name, version)
            registration_tx = mv.tags.get("ario.registration_tx")

            payload: dict = {
                "event_type": "stage_transition",
                "model_name": model_name,
                "model_version": version,
                "from_stage": from_stage,
                "to_stage": to_stage,
            }
            if metadata:
                for k, v in metadata.items():
                    if k in payload:
                        continue
                    payload[k] = v
            payload_bytes = canonical_json(payload)
            payload_hash = hash_data(payload_bytes)

            envelope = self._proof_engine.create_commitment(
                event_type="stage_transition",
                subject={
                    "type": "mlflow_model_version",
                    "name": model_name,
                    "version": str(version),
                },
                payload_bytes=payload_bytes,
                previous_hash=registration_tx or "GENESIS",
            )

            # Defense in depth: wrap upload_proof so a transient
            # Turbo/Arweave outage degrades to signed-only (result=None)
            # rather than aborting the whole _anchor_promotion body via
            # the outer except. Symmetric with anchor() and
            # _anchor_registration.
            if self._anchor.enabled:
                try:
                    result = self._anchor.upload_proof(envelope)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"Promotion upload raised for {model_name}/v{version} "
                        f"({from_stage}->{to_stage}); keeping signed-only proof: {e}"
                    )
                    result = None
            else:
                result = None

            # Write the canonical bytes as an artifact on the source run
            # so verifiers have an immutable witness for check 2. Keyed
            # by the envelope's event_id (NOT just version) — a model
            # version can be promoted multiple times (Staging -> Prod ->
            # Archived), and version-only paths would overwrite each
            # promotion's witness. Verifier resolves via the same
            # event_id (carried in subject + envelope).
            mv = self.get_model_version(model_name, version)
            source_run_id = mv.run_id
            event_id = envelope["event_id"]
            if source_run_id:
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        promotion_dir = os.path.join(
                            tmpdir, "ario", "promotions", event_id,
                        )
                        os.makedirs(promotion_dir)
                        with open(os.path.join(promotion_dir, "payload.json"), "wb") as f:
                            f.write(payload_bytes)
                        with open(os.path.join(promotion_dir, "proof.json"), "w") as f:
                            json.dump(envelope, f, indent=2)
                        # log_artifacts uploads tmpdir/ario/promotions/<event_id>/...
                        # under the run's artifacts/ario/promotions/<event_id>/.
                        self.log_artifacts(
                            source_run_id, os.path.join(tmpdir, "ario"), "ario",
                        )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"Could not write promotion payload artifact for "
                        f"{model_name}/v{version} (event {event_id}): {e}"
                    )

            # Always write ario.promotion_payload_hash — it's the stable
            # pointer to the payload artifact under
            # promotions/<event_id>/payload.json regardless of upload
            # outcome. ario.promotion_tx is only meaningful when the
            # upload succeeded. Symmetric with how registration always
            # writes ario.payload_hash and conditionally writes
            # ario.registration_tx.
            self.set_model_version_tag(
                model_name, version, "ario.promotion_payload_hash", payload_hash,
            )

            if result:
                self.set_model_version_tag(model_name, version, "ario.promotion_tx", result["tx_id"])
                logger.info(
                    f"Promotion {model_name}/v{version} "
                    f"({from_stage}->{to_stage}) anchored: tx={result['tx_id']}"
                )
                self._record_status(
                    "promotion", model_name, version,
                    status="anchored", tx_id=result["tx_id"],
                )
            else:
                # result=None covers three cases that all degrade to
                # signed-only (consistent with anchor() and
                # _anchor_registration): anchor disabled, upload returned
                # no result, or upload raised an exception we caught
                # above. The signed envelope and payload artifact are
                # still in MLflow regardless.
                logger.info(
                    f"Promotion {model_name}/v{version} ({from_stage}->{to_stage}) "
                    f"signed (anchor enabled={self._anchor.enabled})"
                )
                self._record_status(
                    "promotion", model_name, version,
                    status="signed",
                )

        except Exception as e:  # noqa: BLE001 — promotion anchoring runs in a daemon thread; failure logs + records status="failed" without crashing the calling thread
            logger.error(f"Failed to anchor promotion {model_name}/v{version}: {e}")
            self._record_status(
                "promotion", model_name, version,
                status="failed",
                error=str(e),
            )
        finally:
            if done_event is not None:
                done_event.set()
