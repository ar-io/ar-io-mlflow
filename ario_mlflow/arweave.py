"""Arweave upload and retrieval via ar.io Turbo."""

import json
import logging
import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ario_mlflow.proof import canonical_json

logger = logging.getLogger(__name__)

# Where the plugin keeps its auto-generated wallet so the same address is
# reused across sessions. Matches the pattern used by proof.py for signing
# keys (~/.ario-mlflow/keys/).
DEFAULT_WALLET_PATH = os.path.expanduser("~/.ario-mlflow/wallet.json")

# HTTP retry policy for transient gateway failures. Applied to all
# session-based requests (upload, fetch, status). 5xx + 429 are retried
# with exponential backoff; the gateway's ``Retry-After`` header is
# honored when present. 4xx responses other than 429 are NOT retried —
# they indicate a request the gateway already rejected on the merits.
_DEFAULT_MAX_RETRIES = 2  # 1 initial + 2 retries = 3 attempts
_DEFAULT_RETRY_BACKOFF = 0.5  # seconds; doubles each retry: 0.5s, 1.0s
_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)

# Default ordered list of ar.io gateways tried for FETCH operations
# (proof retrieval). Override via the ``gateways`` constructor kwarg or
# the ``ARIO_MLFLOW_GATEWAYS`` env var (comma-separated). Ordering is
# preference: index 0 is tried first; later entries are fallbacks for
# transient gateway outages.
#
# Future swap point: when the AR.IO Network Process gains a Python
# client (or we shell out to the JS wayfinder package), replace
# ``_resolve_gateways`` body with a discovery call and keep this list
# only as the bootstrap fallback.
_DEFAULT_FETCH_GATEWAYS = ("turbo-gateway.com", "ardrive.net")


def _resolve_gateways(
    gateways: list[str] | None,
    gateway_host: str,
) -> list[str]:
    """Resolve the ordered fetch-gateway list.

    Precedence (highest first):

    1. Explicit ``gateways`` kwarg passed to ``ArweaveAnchor``.
    2. ``ARIO_MLFLOW_GATEWAYS`` env var (comma-separated).
    3. Built-in default: ``gateway_host`` first, then any
       :data:`_DEFAULT_FETCH_GATEWAYS` entries not already present.

    Returns a deduplicated list preserving order.
    """
    if gateways is not None:
        candidates = list(gateways)
    elif env := os.environ.get("ARIO_MLFLOW_GATEWAYS"):
        candidates = [g.strip() for g in env.split(",") if g.strip()]
    else:
        candidates = [gateway_host, *_DEFAULT_FETCH_GATEWAYS]

    seen: set[str] = set()
    ordered: list[str] = []
    for g in candidates:
        if g and g not in seen:
            seen.add(g)
            ordered.append(g)
    return ordered

# The three wallet_mode values exposed in logs / tags / reports:
#   user-configured — loaded from a caller-supplied wallet path.
#   persistent      — auto-generated at DEFAULT_WALLET_PATH and reused across runs.
#   ephemeral       — in-memory only (filesystem not writable); rotates every restart.
WALLET_MODE_USER = "user-configured"
WALLET_MODE_PERSISTENT = "persistent"
WALLET_MODE_EPHEMERAL = "ephemeral"

# The funding/upload chain a wallet belongs to. Orthogonal to wallet_mode
# (persistence). Detected from the key's JSON shape, never configured:
#   arweave — an RSA-4096 JWK object (the original/legacy wallet format).
#   solana  — an ed25519 key as a Solana CLI id.json (64-int array) or a
#             base58 secret string. The default for newly generated wallets.
# The upload destination is still Arweave (via Turbo) in both cases; only the
# signer/funding chain differs, so ``ArweaveAnchor`` stays accurately named.
WALLET_TYPE_ARWEAVE = "arweave"
WALLET_TYPE_SOLANA = "solana"

_REQUIRED_JWK_FIELDS = {"kty", "n", "e", "d", "p", "q", "dp", "dq", "qi"}

# A Solana CLI id.json is a 64-int array: 32-byte seed || 32-byte public key.
_SOLANA_ID_JSON_LEN = 64


class WalletLoadError(Exception):
    """A caller-supplied Arweave wallet path could not be loaded.

    Raised when ``ARIO_MLFLOW_ARWEAVE_WALLET`` (or the constructor's
    ``wallet_path`` argument) names a wallet that is missing,
    unreadable, or malformed. The plugin refuses to silently sign with
    an auto-generated wallet under a different identity — operator
    intent must not be silently overridden, since proofs would land
    on-chain under the wrong address with no programmatic signal.
    """


class ArweaveAnchor:
    """Upload proof payloads to Arweave via Turbo SDK."""

    def __init__(
        self,
        wallet_path: str | None = None,
        gateway_host: str = "turbo-gateway.com",
        *,
        gateways: list[str] | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff_factor: float = _DEFAULT_RETRY_BACKOFF,
    ):
        self.gateway_host = gateway_host
        # Ordered fetch-gateway list. ``gateway_host`` retains its role
        # as the "primary" gateway used in returned URLs (the value
        # surfaced to UIs and reports); ``gateways`` is the resilience
        # list iterated when fetches fail.
        self.gateways = _resolve_gateways(gateways, gateway_host)
        self.enabled = False
        self.wallet_mode: str | None = None
        self.wallet_type: str | None = None
        self._signer = None
        self._upload_url = None
        self._token = None
        # Last failure surfaced to callers that get ``None`` from
        # upload_proof / fetch_proof. ``None`` means "no error
        # recorded since the last successful call."
        self.last_error: str | None = None

        # Single session shared across upload, fetch, and status calls.
        # The mounted HTTPAdapter retries 5xx + 429 with exponential
        # backoff; transient gateway failures stop being terminal.
        self._session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=retry_backoff_factor,
            status_forcelist=_RETRY_STATUS_CODES,
            allowed_methods=("GET", "POST", "HEAD"),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        wallet_path = wallet_path or os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", "")

        try:
            from turbo_sdk import ArweaveSigner, SolanaSigner, Turbo
        except ImportError as e:
            logger.warning(f"turbo-sdk not available; Arweave anchoring disabled: {e}")
            return

        # Wallet loading: caller-intent violations (bad ``wallet_path``)
        # raise WalletLoadError and propagate. Default-path failures
        # degrade to ephemeral inside _load_or_create_wallet. The detected
        # ``wallet_type`` selects the signer; the upload destination is
        # Arweave (via Turbo) for both chains.
        secret, wallet_type, mode = self._load_or_create_wallet(wallet_path)

        try:
            if wallet_type == WALLET_TYPE_SOLANA:
                self._signer = SolanaSigner(secret)
            else:
                self._signer = ArweaveSigner(secret)
            turbo = Turbo(self._signer)
            self._upload_url = turbo.upload_url
            self._token = turbo.token  # auto-resolves to "arweave" / "solana"
            self.enabled = True
            self.wallet_mode = mode
            self.wallet_type = wallet_type

            address = self._signer.get_wallet_address()
            if mode == WALLET_MODE_USER:
                logger.info(
                    f"Arweave anchoring enabled (chain={wallet_type}, wallet={address}, "
                    f"mode=user-configured)"
                )
            elif mode == WALLET_MODE_PERSISTENT:
                logger.info(
                    f"Arweave anchoring enabled (chain={wallet_type}, wallet={address}, "
                    f"mode=persistent, path={DEFAULT_WALLET_PATH}) — "
                    f"set ARIO_MLFLOW_ARWEAVE_WALLET to use your own"
                )
            else:
                logger.warning(
                    f"Arweave anchoring enabled (chain={wallet_type}, wallet={address}, "
                    f"mode=ephemeral) — wallet is in-memory only and will rotate on restart. "
                    f"Persistent wallet path {DEFAULT_WALLET_PATH} was not writable."
                )
        except Exception as e:  # noqa: BLE001 — Turbo signer/transport init failure: degrade to disabled
            logger.warning(f"Failed to initialize Turbo signer: {e}")
            self._signer = None
            self.enabled = False

    def _build_default_tags(
        self,
        proof: dict,
        extra_tags: dict[str, str] | None = None,
    ) -> list[dict]:
        """Build the conservative baseline Arweave tag set for a proof.

        Baseline (always-on, derivable from the envelope, non-PII):
        - ``Content-Type``: ``application/json``
        - ``App-Name``: ``ar-io-mlflow``
        - ``App-Version``: installed plugin version
        - ``Event-Type``: from envelope (``training_complete`` /
          ``model_registered`` / ``stage_transition`` / ``prediction``)
        - ``Event-Id``: the envelope's ``event_id``
        - ``Chain-Prev``: the envelope's ``previous_hash``

        Caller-opt-in tags merge in via ``extra_tags``. The plugin never
        auto-writes ``experiment-name``, ``mlflow.source.name``,
        ``git-commit``, or any field that may contain absolute filesystem
        paths or business-sensitive identifiers. See plan Part 3 tag
        policy for the rationale.

        Both pure-commitment envelopes (new shape) and legacy
        record-bearing envelopes (old shape, used by the demo until
        Phase 2) are handled — fields are derived from whichever shape
        the caller passed in.
        """
        # Detect envelope shape. The new pure-commitment envelope has
        # event_type / event_id / previous_hash at the top level. The
        # legacy envelope stores those inside record / record_hash.
        if "payload_hash" in proof:
            event_type = proof.get("event_type", "unknown")
            event_id = proof.get("event_id", "unknown")
            chain_prev = proof.get("previous_hash", "GENESIS")
        else:
            record = proof.get("record", {})
            event_type = record.get("event_type", "unknown")
            event_id = record.get("event_id", record.get("decision_id", "unknown"))
            chain_prev = proof.get("previous_hash", "GENESIS")

        try:
            from importlib.metadata import version
            app_version = version("ar-io-mlflow")
        except Exception:  # noqa: BLE001
            app_version = "unknown"

        baseline = [
            {"name": "Content-Type", "value": "application/json"},
            {"name": "App-Name", "value": "ar-io-mlflow"},
            {"name": "App-Version", "value": app_version},
            {"name": "Event-Type", "value": str(event_type)},
            {"name": "Event-Id", "value": str(event_id)},
            {"name": "Chain-Prev", "value": str(chain_prev)},
        ]

        if extra_tags:
            # Refuse to silently shadow baseline tag keys — caller must
            # use a different key if they want to add something. Avoids
            # confusion where (e.g.) a caller's ``Event-Type`` overwrites
            # the envelope-derived one.
            baseline_keys = {t["name"] for t in baseline}
            for key, value in extra_tags.items():
                if key in baseline_keys:
                    logger.warning(
                        f"extra_tags key {key!r} collides with a baseline "
                        f"tag; ignoring caller-supplied value to keep the "
                        f"envelope-derived value canonical."
                    )
                    continue
                baseline.append({"name": str(key), "value": str(value)})

        return baseline

    @classmethod
    def _detect_and_validate(cls, loaded, source: str):
        """Detect the wallet chain from a parsed JSON value and validate it.

        Chain is a property of the key's JSON *shape*, never configured:

        - a JSON object (``dict``) with the full RSA JWK field set →
          ``arweave`` (returns the JWK dict);
        - a JSON array of 64 byte-ints (a Solana CLI ``id.json``) **or** a
          base58 ``str`` secret → ``solana`` (validated by constructing a
          ``SolanaSigner``, returns the secret as given);
        - anything else (including an incomplete JWK object) →
          :class:`WalletLoadError`.

        Returns ``(secret, wallet_type)``. ``source`` names the wallet for
        error messages.
        """
        if isinstance(loaded, dict):
            if _REQUIRED_JWK_FIELDS.issubset(loaded):
                return loaded, WALLET_TYPE_ARWEAVE
            raise WalletLoadError(
                f"Wallet at {source} is a JSON object but not a complete RSA JWK "
                f"(missing one or more of: {sorted(_REQUIRED_JWK_FIELDS)})"
            )

        if isinstance(loaded, list):
            if len(loaded) != _SOLANA_ID_JSON_LEN or not all(
                isinstance(b, int) and 0 <= b <= 255 for b in loaded
            ):
                raise WalletLoadError(
                    f"Wallet at {source} is a JSON array but not a valid Solana "
                    f"id.json (expected a {_SOLANA_ID_JSON_LEN}-int byte array)"
                )
            secret = loaded
        elif isinstance(loaded, str):
            secret = loaded
        else:
            raise WalletLoadError(
                f"Wallet at {source} is not a recognized format (expected an RSA "
                f"JWK object, a Solana id.json array, or a base58 string)"
            )

        # Validate the Solana secret by attempting to build the signer —
        # catches a bad base58 string or a wrong-length/garbage byte array.
        try:
            from turbo_sdk import SolanaSigner
            SolanaSigner(secret)
        except WalletLoadError:
            raise
        except Exception as e:  # noqa: BLE001 — any SolanaSigner construction failure is a malformed key
            raise WalletLoadError(
                f"Wallet at {source} is not a valid Solana key: {type(e).__name__}: {e}"
            ) from e
        return secret, WALLET_TYPE_SOLANA

    @classmethod
    def _load_or_create_wallet(cls, wallet_path: str) -> tuple[object, str, str]:
        """Return ``(secret, wallet_type, mode)`` for the wallet to use.

        ``wallet_type`` ∈ {``arweave``, ``solana``} is *detected* from the
        key's JSON shape (see :meth:`_detect_and_validate`); ``mode`` ∈
        {``user-configured``, ``persistent``, ``ephemeral``} is the
        persistence axis. Both the env var ``ARIO_MLFLOW_ARWEAVE_WALLET``
        and :data:`DEFAULT_WALLET_PATH` accept *either* chain's key.

        Resolution order:

        1. Caller-supplied ``wallet_path`` (or ``ARIO_MLFLOW_ARWEAVE_WALLET``)
           — the wallet MUST be loadable and well-formed *for its detected
           shape*. Missing file, unreadable file, malformed JSON, an
           incomplete JWK, or an invalid Solana key all raise
           :class:`WalletLoadError` (for **both** chains — the plugin
           never silently substitutes an auto-generated wallet when the
           operator named one).
        2. ``DEFAULT_WALLET_PATH`` — if it already exists, detect its chain
           and reuse it as-is (a legacy RSA ``wallet.json`` is reused
           unchanged). An existing wallet file is **never overwritten**: if
           it is unreadable/unrecognized, the plugin falls back to an
           in-memory wallet rather than clobbering the file.
        3. Fresh install (no wallet present) — generate a **Solana** keypair
           (the default), persist it as a Solana CLI ``id.json`` (64-int
           array) at the path, mode ``persistent``.
        4. If step (3)'s filesystem write fails → in-memory Solana wallet,
           mode ``ephemeral``.
        """
        if wallet_path:
            try:
                with open(wallet_path) as f:
                    loaded = json.load(f)
            except FileNotFoundError as e:
                raise WalletLoadError(
                    f"Arweave wallet path was supplied but file does not exist: "
                    f"{wallet_path}"
                ) from e
            except OSError as e:
                raise WalletLoadError(
                    f"Could not read Arweave wallet at {wallet_path}: {e}"
                ) from e
            except json.JSONDecodeError as e:
                raise WalletLoadError(
                    f"Arweave wallet at {wallet_path} is not valid JSON: {e}"
                ) from e
            secret, wallet_type = cls._detect_and_validate(loaded, wallet_path)
            return secret, wallet_type, WALLET_MODE_USER

        # No user-configured wallet. Reuse a persistent one if present —
        # detecting whichever chain it is — and NEVER overwrite it.
        if os.path.exists(DEFAULT_WALLET_PATH):
            try:
                with open(DEFAULT_WALLET_PATH) as f:
                    loaded = json.load(f)
                secret, wallet_type = cls._detect_and_validate(loaded, DEFAULT_WALLET_PATH)
                return secret, wallet_type, WALLET_MODE_PERSISTENT
            except (OSError, json.JSONDecodeError, WalletLoadError) as e:
                # The file exists but we can't use it. Do NOT overwrite it
                # (no surprise data loss — it may be a wallet format we
                # don't recognize yet). Use an in-memory Solana wallet for
                # this session and tell the operator how to reset.
                logger.warning(
                    f"Existing wallet at {DEFAULT_WALLET_PATH} is unreadable or "
                    f"unrecognized ({e}); not overwriting it. Using an in-memory "
                    f"(ephemeral) Solana wallet for this session — remove the file "
                    f"to provision a fresh persistent wallet."
                )
                return cls._generate_solana_wallet(), WALLET_TYPE_SOLANA, WALLET_MODE_EPHEMERAL

        # Fresh install: generate a Solana wallet (the default) and persist
        # it as a Solana CLI id.json (64-int array) at the single path.
        id_json = cls._generate_solana_wallet()
        try:
            os.makedirs(os.path.dirname(DEFAULT_WALLET_PATH), exist_ok=True)
            with open(DEFAULT_WALLET_PATH, "w") as f:
                json.dump(id_json, f)
            os.chmod(DEFAULT_WALLET_PATH, 0o600)
            logger.info(
                f"Auto-generated Solana wallet at {DEFAULT_WALLET_PATH} — "
                f"back this up or set ARIO_MLFLOW_ARWEAVE_WALLET for production use"
            )
            return id_json, WALLET_TYPE_SOLANA, WALLET_MODE_PERSISTENT
        except OSError as e:
            logger.warning(
                f"Could not persist auto-generated wallet to {DEFAULT_WALLET_PATH}: {e}; "
                f"using in-memory wallet for this session only"
            )
            return id_json, WALLET_TYPE_SOLANA, WALLET_MODE_EPHEMERAL

    @staticmethod
    def _generate_solana_wallet() -> list[int]:
        """Generate a fresh Solana ed25519 keypair as a CLI ``id.json``.

        Uses ``cryptography`` (already a dependency — no new dep). The
        Solana ``id.json`` format is a 64-int byte array: the 32-byte seed
        followed by the 32-byte public key. ``SolanaSigner`` accepts this
        list directly.
        """
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        priv = Ed25519PrivateKey.generate()
        seed = priv.private_bytes_raw()              # 32 bytes
        pub = priv.public_key().public_bytes_raw()   # 32 bytes
        return list(seed + pub)

    @staticmethod
    def _generate_wallet() -> dict:
        """Generate a fresh Arweave RSA-4096 wallet in JWK format."""
        import base64
        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
        pn = private_key.private_numbers()
        pub = pn.public_numbers

        def to_b64(n):
            b = n.to_bytes((n.bit_length() + 7) // 8, "big")
            return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

        return {
            "kty": "RSA",
            "n": to_b64(pub.n),
            "e": to_b64(pub.e),
            "d": to_b64(pn.d),
            "p": to_b64(pn.p),
            "q": to_b64(pn.q),
            "dp": to_b64(pn.dmp1),
            "dq": to_b64(pn.dmq1),
            "qi": to_b64(pn.iqmp),
        }

    def upload_proof(
        self,
        proof: dict,
        tags: list[dict] | None = None,
        extra_tags: dict[str, str] | None = None,
    ) -> dict | None:
        """Upload a proof envelope to Arweave with conservative default tags.

        Args:
            proof: Either a pure-commitment envelope (event_id, event_type,
                payload_hash, previous_hash, ...) or a legacy record-bearing
                envelope (record, record_hash, ...). Both shapes are
                supported during Phase 1; legacy support goes away in
                Phase 2 with the demo refactor.
            tags: Raw Arweave tag list (``[{"name": ..., "value": ...}, ...]``).
                If supplied, replaces the default tag set entirely. Used by
                callers who already know exactly what tags they want.
            extra_tags: Caller-opt-in tags merged with the conservative
                baseline. Use this for ``model-name``, ``mlflow-run-id``,
                ``signer-fingerprint``, or any other indexable metadata
                the caller has decided is safe to expose publicly.
                **Never include PII, internal hostnames, filesystem
                paths, or business-sensitive identifiers** — Arweave tags
                are public, queryable, and permanent.
        """
        self.last_error = None

        if not self.enabled or not self._signer:
            self.last_error = "anchor disabled (turbo-sdk unavailable or wallet unconfigured)"
            return None

        try:
            from turbo_sdk.bundle import create_data, sign

            data_bytes = canonical_json(proof)

            arweave_tags = tags if tags is not None else self._build_default_tags(proof, extra_tags)

            data_item = create_data(bytearray(data_bytes), self._signer, arweave_tags)
            sign(data_item, self._signer)

            url = f"{self._upload_url}/tx/{self._token}"
            raw_data = data_item.get_raw()
            response = self._session.post(
                url,
                data=raw_data,
                headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(raw_data))},
                timeout=60,
            )

            if response.status_code != 200:
                # 4xx (other than 429, which the Retry policy retries)
                # reaches here as a hard failure. Truncate response body
                # to keep logs readable.
                self.last_error = (
                    f"upload returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                logger.error(f"Arweave upload failed: {self.last_error}")
                return None

            receipt = response.json()
            tx_id = receipt["id"]
            logger.info(f"Uploaded to Arweave: tx_id={tx_id}")
            return {"tx_id": tx_id, "url": f"https://{self.gateway_host}/{tx_id}", "receipt": receipt}

        except requests.exceptions.RequestException as e:
            # Covers ConnectionError, Timeout, and RetryError (raised
            # when urllib3's Retry policy exhausts).
            self.last_error = f"upload network/HTTP error: {type(e).__name__}: {e}"
            logger.error(f"Arweave upload failed: {self.last_error}")
            return None
        except Exception as e:  # noqa: BLE001 — preserve None-return contract for unexpected failures; full traceback logged
            self.last_error = f"upload unexpected error: {type(e).__name__}: {e}"
            logger.error(f"Arweave upload failed: {self.last_error}", exc_info=True)
            return None

    def fetch_proof(self, tx_id: str) -> dict | None:
        """Fetch a proof envelope by Arweave TX ID.

        Iterates :attr:`gateways` in order; on transient HTTP/network
        errors against one gateway, falls back to the next. Returns the
        parsed JSON on first success, or ``None`` if every gateway
        failed (with the failure trail recorded in ``self.last_error``).
        """
        self.last_error = None
        if not self.gateways:
            self.last_error = "no fetch gateways configured"
            logger.error(self.last_error)
            return None

        errors: list[str] = []
        for gateway in self.gateways:
            url = f"https://{gateway}/raw/{tx_id}"
            try:
                resp = self._session.get(url, timeout=30)
                resp.raise_for_status()
                # ValueError from resp.json() (gateway returned 200 with
                # non-JSON body) is treated as a gateway failure, not a
                # caller-side bug — fall over to the next gateway.
                parsed = resp.json()
                if gateway != self.gateways[0]:
                    # Surface the fact that we failed over so ops can
                    # see it in logs without parsing every request.
                    logger.info(
                        f"Fetched {tx_id} from fallback gateway {gateway} "
                        f"after primary {self.gateways[0]} failed"
                    )
                return parsed
            except (requests.exceptions.RequestException, ValueError) as e:
                errors.append(f"{gateway}: {type(e).__name__}: {e}")
                logger.warning(
                    f"Gateway {gateway} failed for tx {tx_id}: {type(e).__name__}: {e}"
                )
                continue

        self.last_error = (
            f"fetch failed across {len(self.gateways)} gateway(s) "
            f"({', '.join(self.gateways)}): {' | '.join(errors)}"
        )
        logger.error(f"All gateways failed for tx {tx_id}: {self.last_error}")
        return None

    def check_status(self, tx_id: str) -> dict:
        """Query Turbo's bundler-status endpoint for ``tx_id``.

        Single-endpoint by design: this hits Turbo's internal status
        service (``turbo.ardrive.io/tx/<id>/status``), not a generic
        ar.io gateway. Multi-gateway fallback isn't applicable —
        finalization status is Turbo-specific. Retries against the same
        endpoint on transient errors via the shared session.
        """
        try:
            resp = self._session.get(
                f"https://turbo.ardrive.io/tx/{tx_id}/status", timeout=10
            )
            if resp.status_code == 200:
                # ValueError if the response isn't JSON — handled below
                # alongside transport errors so callers always get a dict.
                data = resp.json()
                return {"status": data.get("status", "UNKNOWN"), "info": data.get("info")}
            return {"status": "NOT_FOUND"}
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.error(f"Failed to check Turbo status for {tx_id}: {type(e).__name__}: {e}")
            return {"status": "UNKNOWN"}
