# `VerifiedModel` — agent verify-status gating

> **Status: shipped (v1.2 Lane E).** Consumer of the ar-io-agent wire contract
> [`verify-status-api.md`](https://github.com/ar-io/ar-io-agent/blob/main/docs/verify-status-api.md)
> (v1). The contract is binding: this plugin branches on HTTP status codes and the
> structured `outcome` field only, never on error-message strings.

`VerifiedModel` has always re-hashed model artifacts against the anchored
`ario.artifact_hash` before loading pyfunc code (raising `IntegrityError` on
mismatch). As of v1.2 it can **additionally** consult an ar-io-agent watching the
model's files and refuse to expose a model whose covering asset is tampered,
missing, stale, or unknown — turning the agent's anchored evidence into a
load-time gate.

## Quick start

```python
from ario_mlflow import VerifiedModel, VerifyStatusClient

client = VerifyStatusClient(
    "http://127.0.0.1:9847",                 # the agent's management port
    secret=open("/var/lib/ario-agent/management-secret").read().strip(),
)

model = VerifiedModel(
    "models:/fraud-detector@production",
    asset_id="fraud-model",                  # the asset_id from the agent's policy.yaml
    verify_status_client=client,
    on_failure="fail_closed",
)
model.predict(X)
```

All three keyword arguments are optional — omit them and `VerifiedModel`
behaves exactly as before (artifact integrity check only).

## Gate ordering

1. **Agent verify-status** — one local HTTP read of agent state on the
   loopback management port. Never triggers a re-verify, never makes a
   network call beyond the agent.
2. **Artifact integrity** — re-hash artifacts vs `ario.artifact_hash`
   (downloads artifacts; this is why the cheap agent check runs first).
3. **pyfunc load** — user code executes only after both gates pass.

The gate runs at **load time only**. A tamper the agent detects *after* the
constructor returns is not re-checked on `predict()` — per-predict re-checking
(with the contract's 10–30s caching guidance) is a planned follow-up. Restart
or reconstruct `VerifiedModel` to re-gate.

## Failure-mode matrix

Mapping per contract §9.1, implemented verbatim:

| `outcome` | `on_failure="raise"` exception | `fail_closed` behavior | `fail_open` behavior |
|---|---|---|---|
| `verified` (stale=false) | (no error — proceed) | proceed | proceed |
| `verified` (stale=true) | `AssetStaleError` | raise | log + proceed |
| `tampered` | `AssetTamperedError` | raise | log + proceed |
| `missing` | `AssetMissingError` | raise | log + proceed |
| `unavailable` | `AssetStaleError` (treat as stale) | raise | log + proceed |
| `unknown` | `AssetUnknownError` | raise | log + proceed |

Transport-level failures follow the same `on_failure` policy:

| Condition | Exception |
|---|---|
| HTTP 401 (bad secret / API key) | `VerifyStatusAuthError` |
| HTTP 404 (`asset_id` not in the agent's policy) | `VerifyStatusUnknownAssetError` |
| HTTP 503 (license gate: plan lacks block enforcement) | `VerifyStatusLicenseError` — carries `upgrade_url`; a purchasing signal, not a verification failure |
| Network failure / malformed body / other non-200 | `VerifyStatusTransportError` |

`"raise"` and `"fail_closed"` are identical in behavior — the latter name reads
better in production configs. **`"fail_open"` logs at WARN** with structured
fields (`extra["ario_verify_status"]`: `asset_id`, `error`, `outcome`, `stale`,
`policy_hash`, `current_tx_id`) and proceeds; route that logger to your SIEM. A
model load that silently bypasses verification is a regulatory liability, so
the line is loud and machine-keyable.

### Exception hierarchy

```
AssetVerificationError                 ← catch both gates with one clause
├── IntegrityError                     (artifact re-hash; ALWAYS raises — on_failure does not apply)
└── VerifyStatusError                  ← anything from the verify-status path
    ├── AssetTamperedError / AssetMissingError / AssetStaleError / AssetUnknownError
    ├── VerifyStatusAuthError
    ├── VerifyStatusUnknownAssetError
    └── VerifyStatusTransportError
        └── VerifyStatusLicenseError   (.upgrade_url)
```

Every exception carries `.asset_id` and `.status` (the raw `VerifyStatus`
response, when one was received).

## Deployment topology

**Same host (loopback).** The agent's management port binds `127.0.0.1:9847`
and is deliberately not exposed beyond the host. Run your model server on the
agent's host and authenticate with the management secret
(`<state-dir>/management-secret`, or the `ARIO_AGENT_MANAGEMENT_SECRET` env
override in k8s):

```python
VerifyStatusClient("http://127.0.0.1:9847", secret=...)
```

**Cross-host: not supported in v1.2.** The agent's verify-status endpoint is
loopback-only by design, and **no api-guard proxy route exists** — an earlier
draft of the contract proposed one, but it was withdrawn before
implementation (it would have required api-guard→agent connectivity that
contradicts the loopback-bind invariant; see `verify-status-api.md` v1.1
§9.2 in `ar-io-agent`). Run the consumer on the agent's host. The client's
`api_key=` constructor mode is a forward-compatibility reservation for a
possible future hosted topology; it currently has no server-side
counterpart.

**License gating.** Plans without block enforcement receive
`503 license required` → `VerifyStatusLicenseError` (with `upgrade_url`).
The **agent itself** emits the 503, driven by entitlement state api-guard
delivers on register/heartbeat responses — fail-open when that state is
absent, so an unlicensed or offline api-guard never bricks verify-status.
The client branches on the status code alone.

## Freshness semantics (`max_age` vs `stale`)

Trust the agent's `stale` boolean, not your own clock math. `max_age` is the
*nominal* policy reverify interval (stable, cacheable, changes only with
`policy_hash`); `stale` is server-computed with the same due-work math the
agent's scheduler uses, including unavailability backoff and per-asset jitter
(contract §6). `VerifiedModel` always fetches a fresh status at load time.
`VerifyStatusClient.get(asset_id, max_cache_age=...)` offers opt-in
monotonic-clock caching for hot-path consumers (10–30s per contract §9.2).

## Relationship to the artifact integrity check

The two gates are complementary, not redundant: the agent watches the
**deployed files** named in its policy (its evidence is anchored on Arweave);
the integrity check re-hashes the **MLflow registry artifacts** behind the
`models:/` URI. Tampering with the serving copy trips the agent gate; tampering
with the registry copy trips `IntegrityError`. `on_failure` applies only to the
agent gate — `IntegrityError` always raises.
