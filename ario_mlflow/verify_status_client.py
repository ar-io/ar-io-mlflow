"""HTTP client for the agent's ``GET /v1/verify-status/<asset_id>`` endpoint.

Implements the consumer side of `ar-io-agent/docs/verify-status-api.md`
(v1, contract draft 2026-06-10). The endpoint is loopback-only by design —
the agent binds ``127.0.0.1:9847`` and there is **no api-guard proxy
route**: a cross-host proxy was proposed early in v1.2 design and
withdrawn before implementation (it would have required api-guard→agent
connectivity that contradicts the loopback-bind invariant; see
``verify-status-api.md`` v1.1 §9.2).

The client therefore has **one supported deployment form** today:

- **Same-host management port** (``http://127.0.0.1:9847``) — pass
  ``secret=`` (the agent's ``<state-dir>/management-secret`` value);
  sent as ``X-Ario-Management-Secret``.

The ``api_key=`` constructor mode is a **documented forward-compatibility
reservation**: it currently has no server-side counterpart and the agent
itself never honors a ``Authorization: Bearer`` header. The constructor
still accepts it (matching contract §2.1's "exactly one of" shape) so a
future hosted topology — if one ever lands — can ship without breaking
the consumer signature. Code paths that construct with ``api_key=`` will
get a transport error from any agent today.

Contract discipline baked in:

- Branch on HTTP status codes only — never on English error strings (§10).
- Unrecognized ``outcome`` values are normalized to ``"unknown"`` (§10).
- Unknown response fields are ignored (§10).
- ``max_age`` + ``policy_hash`` are cacheable; ``outcome`` + ``stale`` are
  not — by default every :meth:`VerifyStatusClient.get` performs a fresh
  HTTP request. The optional ``max_cache_age`` argument exists for
  hot-path consumers per §9.2 (10–30s is the contract's guidance);
  staleness math uses ``time.monotonic()``, never the agent's clock.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from urllib.parse import quote

import requests

from ario_mlflow.errors import (
    VerifyStatusAuthError,
    VerifyStatusLicenseError,
    VerifyStatusTransportError,
    VerifyStatusUnknownAssetError,
)

logger = logging.getLogger(__name__)

#: The closed v1 outcome vocabulary (contract §10).
OUTCOMES = frozenset({"verified", "tampered", "missing", "unavailable", "unknown"})


@dataclass(frozen=True)
class VerifyStatus:
    """One verify-status response, mirroring contract §3.

    ``outcome`` is always one of :data:`OUTCOMES` — unrecognized server
    values are normalized to ``"unknown"`` at parse time (§10).
    """

    asset_id: str
    tenant_id: str
    agent_id: str
    outcome: str
    last_verified_at: str | None
    max_age: int
    stale: bool
    policy_hash: str
    current_tx_id: str | None


def _parse_status(asset_id: str, body: dict) -> VerifyStatus:
    """Build a :class:`VerifyStatus` from a 200-OK JSON body.

    Unknown fields are ignored; missing required fields raise
    :class:`VerifyStatusTransportError` (a 200 we cannot parse is a
    transport-level failure, not a verification verdict).
    """
    try:
        outcome = body["outcome"]
        status = VerifyStatus(
            asset_id=body["asset_id"],
            tenant_id=body["tenant_id"],
            agent_id=body["agent_id"],
            outcome=outcome if outcome in OUTCOMES else "unknown",
            last_verified_at=body["last_verified_at"],
            max_age=int(body["max_age"]),
            stale=bool(body["stale"]),
            policy_hash=body["policy_hash"],
            current_tx_id=body["current_tx_id"],
        )
    except (KeyError, TypeError, ValueError) as e:
        raise VerifyStatusTransportError(
            f"malformed verify-status response for asset {asset_id!r}: {e!r}",
            asset_id=asset_id,
        ) from e
    if outcome not in OUTCOMES:
        logger.warning(
            f"verify-status returned unrecognized outcome {outcome!r} for "
            f"asset {asset_id!r}; treating as 'unknown' per contract §10"
        )
    return status


class VerifyStatusClient:
    """Client for ``GET /v1/verify-status/<asset_id>``.

    Args:
        base_url: ``http://127.0.0.1:9847`` for the same-host management
            port (the only deployment form with a server-side counterpart
            today; see the module docstring).
        secret: The agent's management secret. Sent as
            ``X-Ario-Management-Secret``.
        api_key: Forward-compatibility reservation — passed as
            ``Authorization: Bearer <key>``, but no agent endpoint honors
            it today; calls will fail at transport. Reserved so a future
            hosted topology can ship without breaking this signature.
        timeout: Per-request timeout in seconds.
        session: Optional ``requests.Session`` override (testing / pooling).

    Exactly one of ``secret`` / ``api_key`` must be provided — the
    contract has no anonymous mode (contract §2.1).
    """

    def __init__(
        self,
        base_url: str,
        secret: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = 5.0,
        session: requests.Session | None = None,
    ):
        if (secret is None) == (api_key is None):
            raise ValueError(
                "exactly one of secret= (management port) or api_key= "
                "(forward-compat reservation — no server-side counterpart today) "
                "is required"
            )
        self._base_url = base_url.rstrip("/")
        if secret is not None:
            self._headers = {"X-Ario-Management-Secret": secret}
        else:
            self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout
        self._session = session or requests.Session()
        # asset_id -> (time.monotonic() at receipt, VerifyStatus).
        # max_age/policy_hash ride along on the cached instance; outcome/stale
        # are only served from here when the caller opts in via max_cache_age.
        self._cache: dict[str, tuple[float, VerifyStatus]] = {}

    def get(self, asset_id: str, *, max_cache_age: float | None = None) -> VerifyStatus:
        """Fetch the verification state for ``asset_id``.

        Args:
            asset_id: The operator-chosen policy asset_id. Percent-encoded
                for the path automatically (contract §2.2).
            max_cache_age: When set, a response received within the last
                ``max_cache_age`` seconds (monotonic) is returned without
                an HTTP request — the §9.2 hot-path pattern. Default
                ``None`` always fetches fresh: ``outcome`` and ``stale``
                are not cacheable for gating decisions (§9.1).

        Raises:
            VerifyStatusAuthError: HTTP 401.
            VerifyStatusUnknownAssetError: HTTP 404 (not in policy).
            VerifyStatusLicenseError: HTTP 503 (the license gate refused
                the request).
            VerifyStatusTransportError: network failure, malformed body,
                or any other non-200 status.
        """
        if max_cache_age is not None:
            cached = self._cache.get(asset_id)
            if cached is not None and time.monotonic() - cached[0] < max_cache_age:
                return cached[1]

        url = f"{self._base_url}/v1/verify-status/{quote(asset_id, safe='')}"
        try:
            # allow_redirects=False is a CREDENTIAL-SAFETY measure, not a
            # nicety: requests preserves custom headers (our
            # X-Ario-Management-Secret) across cross-host redirects — only
            # Authorization is auto-stripped — so a 3xx from a compromised
            # or misconfigured upstream would otherwise hand the management
            # secret to an attacker-chosen host. A verify-status lookup is a
            # fixed JSON endpoint that never legitimately redirects, so any
            # 3xx is treated as a transport error below.
            resp = self._session.get(
                url,
                headers=self._headers,
                timeout=self._timeout,
                allow_redirects=False,
            )
        except requests.RequestException as e:
            raise VerifyStatusTransportError(
                f"verify-status request failed for asset {asset_id!r}: {e}",
                asset_id=asset_id,
            ) from e

        if resp.status_code == 200:
            try:
                body = resp.json()
            except ValueError as e:
                raise VerifyStatusTransportError(
                    f"verify-status returned non-JSON 200 body for asset {asset_id!r}",
                    asset_id=asset_id,
                ) from e
            status = _parse_status(asset_id, body)
            self._cache[asset_id] = (time.monotonic(), status)
            return status

        if resp.status_code == 401:
            raise VerifyStatusAuthError(
                f"verify-status rejected credentials for asset {asset_id!r} (HTTP 401)",
                asset_id=asset_id,
            )
        if resp.status_code == 404:
            raise VerifyStatusUnknownAssetError(
                f"asset {asset_id!r} is not in the agent's current policy (HTTP 404)",
                asset_id=asset_id,
            )
        if resp.status_code == 503:
            upgrade_url = None
            try:
                upgrade_url = resp.json().get("upgrade_url")
            except ValueError:
                pass
            message = (
                f"verify-status for asset {asset_id!r} was refused by the license "
                f"gate: the active plan does not include block enforcement (HTTP 503)"
            )
            if upgrade_url:
                message += f"; upgrade at {upgrade_url}"
            raise VerifyStatusLicenseError(
                message, asset_id=asset_id, upgrade_url=upgrade_url
            )
        raise VerifyStatusTransportError(
            f"verify-status returned HTTP {resp.status_code} for asset "
            f"{asset_id!r}: {resp.text[:200]}",
            asset_id=asset_id,
        )
