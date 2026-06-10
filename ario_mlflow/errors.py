"""Typed exceptions for agent verify-status gating.

One coherent family so consumers can catch broadly or precisely:

- ``AssetVerificationError`` — root of every "the load was refused"
  failure. :class:`ario_mlflow.model.IntegrityError` (the artifact
  re-hash gate) also subclasses it, so ``except AssetVerificationError``
  catches both gates with one clause.
- ``VerifyStatusError`` — anything arising from the agent's
  ``/v1/verify-status`` consultation, whether the asset failed
  verification or the agent could not be asked.

Hierarchy::

    AssetVerificationError
    ├── IntegrityError                  (ario_mlflow.model — artifact re-hash)
    └── VerifyStatusError
        ├── AssetTamperedError          outcome=tampered
        ├── AssetMissingError           outcome=missing
        ├── AssetStaleError             verified+stale, or outcome=unavailable
        ├── AssetUnknownError           outcome=unknown
        ├── VerifyStatusAuthError       HTTP 401
        ├── VerifyStatusUnknownAssetError  HTTP 404 (asset_id not in policy)
        └── VerifyStatusTransportError  network failure / other non-200
            └── VerifyStatusLicenseError  HTTP 503 license required (api-guard)

The outcome→exception mapping implements `verify-status-api.md` §9.1
verbatim (see :func:`exception_for_status`). Exceptions never embed
parsed English error strings from the server — branching is on HTTP
status codes and the structured ``outcome`` field only (contract §10).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type-only import, avoids a cycle
    from ario_mlflow.verify_status_client import VerifyStatus


class AssetVerificationError(Exception):
    """Root of the load-time verification failure family.

    Both gates raise subclasses of this: the agent verify-status gate
    (everything in this module) and the artifact integrity gate
    (:class:`ario_mlflow.model.IntegrityError`).
    """


class VerifyStatusError(AssetVerificationError):
    """Base for all failures arising from the agent verify-status check.

    Attributes:
        asset_id: The policy asset_id that was being checked. ``None``
            only when the failure happened before an asset was in play.
        status: The raw :class:`~ario_mlflow.verify_status_client.VerifyStatus`
            response, when one was received. ``None`` for transport-level
            failures where no valid response body exists.
    """

    def __init__(
        self,
        message: str,
        *,
        asset_id: str | None = None,
        status: "VerifyStatus | None" = None,
    ):
        super().__init__(message)
        self.asset_id = asset_id
        self.status = status


class AssetTamperedError(VerifyStatusError):
    """The agent's last verification observed bytes that do not match the baseline."""


class AssetMissingError(VerifyStatusError):
    """The asset has been unavailable long enough that the agent declared it missing."""


class AssetStaleError(VerifyStatusError):
    """The verification state is too old to trust (stale), or the asset is
    transiently unavailable (treated as stale per contract §9.1)."""


class AssetUnknownError(VerifyStatusError):
    """The agent knows the asset_id but has no verification state for it yet."""


class VerifyStatusAuthError(VerifyStatusError):
    """The endpoint rejected our credentials (HTTP 401)."""


class VerifyStatusUnknownAssetError(VerifyStatusError):
    """The asset_id is not in the agent's current policy (HTTP 404).

    Distinct from :class:`AssetUnknownError`: 404 means "no record of this
    asset_id in policy"; ``outcome=unknown`` means "in policy, not yet
    verified" (contract §8.7).
    """


class VerifyStatusTransportError(VerifyStatusError):
    """The agent (or proxy) could not be asked, or answered unusably.

    Covers network failures, malformed response bodies, and any HTTP
    status outside the contract's enumerated set. Per contract §9.1,
    consumers apply the same ``on_failure`` policy to transport errors
    as to verification failures.
    """


class VerifyStatusLicenseError(VerifyStatusTransportError):
    """api-guard refused the request because the tenant's plan does not
    include block enforcement (HTTP 503, contract §7).

    This is a purchasing signal, not a verification failure — the asset
    was never checked. ``upgrade_url`` carries api-guard's plan upgrade
    link when the response body included one.
    """

    def __init__(
        self,
        message: str,
        *,
        asset_id: str | None = None,
        status: "VerifyStatus | None" = None,
        upgrade_url: str | None = None,
    ):
        super().__init__(message, asset_id=asset_id, status=status)
        self.upgrade_url = upgrade_url


def exception_for_status(status: "VerifyStatus") -> VerifyStatusError | None:
    """Map a 200-OK verify-status response to the §9.1 gate exception.

    Returns ``None`` exactly when the load may proceed: ``outcome=verified``
    and not stale. Unrecognized outcomes were already normalized to
    ``unknown`` by the client (contract §10), so the fall-through here is
    defensive only.
    """
    if status.outcome == "verified":
        if status.stale:
            return AssetStaleError(
                f"asset {status.asset_id!r} verification state is stale "
                f"(last_verified_at={status.last_verified_at}, max_age={status.max_age}s)",
                asset_id=status.asset_id,
                status=status,
            )
        return None
    if status.outcome == "tampered":
        return AssetTamperedError(
            f"asset {status.asset_id!r} failed verification: observed bytes do not "
            f"match the baseline (anchored evidence: tx {status.current_tx_id})",
            asset_id=status.asset_id,
            status=status,
        )
    if status.outcome == "missing":
        return AssetMissingError(
            f"asset {status.asset_id!r} has been declared missing by the agent "
            f"(anchored evidence: tx {status.current_tx_id})",
            asset_id=status.asset_id,
            status=status,
        )
    if status.outcome == "unavailable":
        return AssetStaleError(
            f"asset {status.asset_id!r} is presently unavailable to the agent "
            f"(transient; treated as stale per contract §9.1)",
            asset_id=status.asset_id,
            status=status,
        )
    # "unknown" plus any unrecognized outcome normalized by the client.
    return AssetUnknownError(
        f"asset {status.asset_id!r} has no verification state yet "
        f"(outcome={status.outcome!r})",
        asset_id=status.asset_id,
        status=status,
    )
