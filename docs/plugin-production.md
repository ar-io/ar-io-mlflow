# Putting `ar-io-mlflow` into production

The plugin is alpha — this guide covers the operational gaps the README
doesn't go into. Read alongside [`plugin-threat-model.md`](plugin-threat-model.md)
before going live.

## Wallet management

The auto-generated wallet at `~/.ario-mlflow/wallet.json` is fine for
local development and proofs-of-concept. For anything else, **generate a
dedicated wallet per environment** and treat it like any other production
secret.

### Per-environment wallets

A typical setup:

| Environment | Wallet | Set via |
|---|---|---|
| Local dev | Auto-generated, default path | (no env var) |
| CI / dev | Shared dev wallet, low balance | `ARIO_MLFLOW_ARWEAVE_WALLET=/secrets/dev-wallet.json` |
| Staging | Dedicated staging wallet | `ARIO_MLFLOW_ARWEAVE_WALLET=/secrets/staging-wallet.json` |
| Production | Dedicated prod wallet, monitored | `ARIO_MLFLOW_ARWEAVE_WALLET=/secrets/prod-wallet.json` |

Mixing environments on one wallet means a dev mistake can pollute prod's
on-chain history. Per-environment wallets isolate that.

### Wallet chain — Solana by default

Newly generated wallets are **Solana** (ed25519), written as a Solana CLI
`id.json` at the wallet path. **Arweave RSA wallets remain fully supported**;
the plugin detects the chain from the key's JSON shape (an RSA JWK object →
Arweave; a 64-int `id.json` array or base58 string → Solana). Ops consequences:

- A wallet file already on disk is **reused as-is and never overwritten** — a
  pre-Solana RSA `wallet.json` keeps its address, no migration needed.
- To pin a chain, point `ARIO_MLFLOW_ARWEAVE_WALLET` at that chain's key.
- A malformed *caller-supplied* wallet raises `WalletLoadError` (for either
  chain) — the plugin never silently substitutes a different identity.
- The anchor destination is Arweave (via Turbo) regardless of funding chain;
  the env var keeps its legacy name for backward compatibility.

### Loading the wallet from a secrets manager

`ARIO_MLFLOW_ARWEAVE_WALLET` is a file path, not the key contents (the file
holds a Solana `id.json` / base58 secret or an Arweave RSA JWK). Wire your
secrets manager to materialize the file at runtime:

**HashiCorp Vault** (Vault Agent sidecar pattern):
```hcl
template {
  source      = "/etc/vault-templates/wallet.json.tmpl"
  destination = "/secrets/wallet.json"
  perms       = 0400
}
```
Set `ARIO_MLFLOW_ARWEAVE_WALLET=/secrets/wallet.json` in the workload.

**AWS Secrets Manager** (init container pattern in K8s):
```yaml
initContainers:
  - name: fetch-wallet
    image: amazon/aws-cli
    command: ["sh", "-c"]
    args:
      - aws secretsmanager get-secret-value --secret-id arweave/prod-wallet
        --query SecretString --output text > /secrets/wallet.json &&
        chmod 0400 /secrets/wallet.json
    volumeMounts:
      - name: wallet
        mountPath: /secrets
```
Same env var pointing at the materialized file.

**Kubernetes Secret** (simplest, less rotation control):
Mount a `Secret` containing the wallet key (Solana `id.json` or RSA JWK) as a
file, point the env var at it.

### Wallet permissions

The plugin doesn't enforce file permissions on caller-supplied wallets
(it does set `0o600` on auto-generated ones). Set permissions explicitly
to `0400` on the materialized file, and ensure the workload's user owns
it.

### Key rotation

Ed25519 signing keys are independent of the funding wallet. Rotate them
separately:

- **Funding-wallet rotation** (Solana or Arweave): generate a new wallet
  key, fund it if needed, update the secret, restart workloads. New uploads
  are funded by the new wallet address; existing proofs remain valid
  (verification depends on the envelope's embedded Ed25519 `public_key`,
  never on the funding wallet).
- **Ed25519 signing key rotation**: similar — new key takes over for
  new proofs; old proofs remain valid against the public key embedded
  in their envelope. Document the rotation in your security log so
  auditors don't see the address change as suspicious.

After any rotation, publish the new public key through your
trusted-identity channel (signed announcement, key transparency, etc.)
so verifiers can bind the new key to your organization.

## CI/CD patterns

### Training pipelines (GitHub Actions, GitLab CI, Jenkins)

The wallet needs to live somewhere CI can read it. Patterns:

1. **CI-scoped secret** (most common): store the wallet key as a CI secret,
   write to a temp file at job start, set `ARIO_MLFLOW_ARWEAVE_WALLET`
   to that path. Cleanup happens automatically when the runner is torn
   down.
2. **OIDC-to-cloud-secrets** (cleaner): use GitHub OIDC / GitLab JWT to
   assume an IAM role, fetch from your secrets manager, materialize as
   above. No long-lived wallet in CI config.
3. **Dedicated CI wallet** (operational simplicity): one wallet for all
   CI runs, low balance, monitored. If compromised, rotate without
   touching prod.

Don't share the same wallet across CI and production training. Even if
both are signing real proofs, mixing the chain on one wallet makes
forensics harder.

### Inference services

`VerifiedModel.predict()` writes to the wallet on every prediction
(asynchronously). Workloads need either:

- Wallet file mounted into every replica (K8s Secret, Vault sidecar)
- Or the wallet key materialized to a file from an env var / secret — but
  file-based (`ARIO_MLFLOW_ARWEAVE_WALLET` as a path) is the supported path

If you scale a service horizontally, every replica signs independently
with the same wallet. That's fine for chain semantics (predictions
chain to the model version's `ario.registration_tx`, not to each
other) and avoids cross-replica coordination.

### Agent verify-status gate (runtime tamper detection)

If the model's deployed files are watched by the sister
[`ar-io-agent`](https://github.com/ar-io/ar-io-agent) daemon, pair
`VerifiedModel` with a `VerifyStatusClient` so the agent's verdict gates
load and (optionally) every `predict()`. The gate is **load-time only by
default**; pass `recheck_per_predict=True` (and `recheck_max_cache_age=`
for the contract §9.2 hot-path cache) to extend it into the predict path.

Operational shape:

- **Co-locate the model server and the agent** on the same host — the
  agent's `/v1/verify-status/<asset_id>` endpoint is loopback-only (binds
  `127.0.0.1:9847` by design). There is no cross-host proxy form.
- **Provision the management secret** as a file mount from the agent's
  `<state-dir>/management-secret`, or read it from
  `ARIO_AGENT_MANAGEMENT_SECRET` if your deployment plumbs it in via env.
- **Choose `on_failure` deliberately.** `"fail_closed"` is the regulatory
  default — a tamper / missing / stale verdict refuses inference.
  `"fail_open"` proceeds **but logs a structured WARN** with
  `extra["ario_verify_status"]` including `phase` (`load` / `predict`),
  `asset_id`, `outcome`, `stale`, `policy_hash`, `current_tx_id`. Route
  that logger to your SIEM; a silent bypass is a regulatory liability.
- **License gating.** Plans without block enforcement receive a `503` →
  `VerifyStatusLicenseError` (carries `upgrade_url`). The agent emits the
  503 based on api-guard entitlement state, fail-open when that state is
  absent — an unlicensed or offline api-guard never bricks the gate.

Full failure-mode matrix in [`docs/verified-model.md`](verified-model.md).

## Monitoring and alerting

The plugin doesn't ship monitoring hooks. Wire your own:

### What to alert on

| Signal | Where | Why it matters |
|---|---|---|
| `ario.verify_status: signed` (not `anchored`) on training runs | MLflow tag query | Anchoring fell back to local-sign; Arweave upload failed (gateway down, balance issue, network) |
| `IntegrityError` in inference logs | App logs | T1 — model artifact swap detected; **page security ops** |
| `WalletLoadError` at service startup | App logs | T6 — wallet path misconfigured; service can't sign |
| `last_error` populated on `ArweaveAnchor` instance | Programmatic check | Failure cause for `None` returns; useful for dashboards |
| Anchor latency p99 spike | OTel metrics around `anchor()` calls | Gateway slowdown |
| `ario.attestation_level` < expected for anchored proofs | Tag query, scheduled job | ar.io Verify maturity stalled — proof might be propagating slowly |
| Wallet balance below threshold | External cron against Turbo balance API | Avoid running out of credit if you exceed free tier |

### Where MLflow tags live

Anything in `ario.*` tags is queryable via MLflow's standard tag-search
API. A scheduled job that runs `mlflow.search_runs(filter_string="tags.ario.verify_status = 'signed'")`
catches uploads that should have anchored but didn't.

### Dashboards

A useful dashboard panel set:

- Anchor success rate (anchored / total) over time
- Median + p99 latency for `anchor()`
- Count of `IntegrityError` events (should be zero)
- Distribution of `attestation_level` for the last 24h of proofs (should
  trend up to 3 over time)
- Wallet balance trend

## Operational runbooks

### "Anchoring is failing for everything"

1. Check `last_error` on a fresh `ArweaveAnchor` instance — single source
   of truth for the cause.
2. Check Turbo gateway status: `https://turbo.ardrive.io/tx/<recent-tx>/status`.
3. Check wallet balance — if you exceed Turbo's free tier and the wallet
   is empty, every upload will reject.
4. Check network egress to `turbo-gateway.com` and your fallback gateways.
5. Re-run a known-good training job; if it anchors, the issue was
   transient. If not, escalate.

Failed anchoring degrades to signed-only — your MLflow runs still
succeed, you just don't have an Arweave TX. Re-anchor by re-running
training with the same artifacts (the proof envelope will have a new
`event_id` and `signed_at`, but the same `payload_hash`).

### "Verification fails for an old proof"

1. `Proof Found = FAIL`: the TX isn't on the gateway you're querying.
   Try `ARIO_MLFLOW_GATEWAYS=arweave.net,turbo-gateway.com` to force a
   different fetch path.
2. `Record Matches = FAIL`: either MLflow tampering (the canonical
   payload's bytes don't hash to the envelope's `payload_hash`), or the
   live re-derivation surface drifted. Compare `ario/payload.json`
   directly against the envelope's hash by hand to localize.
3. `Signature Confirmed = FAIL`: the public key in the envelope doesn't
   match — possibly an envelope was rewritten in transit (impossible on
   Arweave's content-addressing) or a malicious gateway returned forged
   bytes. Re-fetch from a different gateway.

### "I rotated my signing key — old proofs broke"

They didn't. Each envelope embeds the public key it was signed with;
verification uses that key, not the current one. If verification fails
post-rotation, it's not the rotation — re-investigate via the runbook
above.

## Hardening checklist before production

- [ ] Dedicated wallet per environment, materialized from a secrets manager
- [ ] `ARIO_MLFLOW_SIGNING_KEY` set explicitly (don't rely on auto-generated key)
- [ ] `ARIO_MLFLOW_ARIO_VERIFY_URL` configured (otherwise the attestation row never runs)
- [ ] `ARIO_MLFLOW_GATEWAYS` configured with at least two operators you've verified
- [ ] Monitoring on `verify_status = signed` and `IntegrityError`
- [ ] Wallet balance alerting (if you expect to exceed free tier)
- [ ] Public key published through your trusted-identity channel
- [ ] Threat model reviewed by your security team — see
  [`plugin-threat-model.md`](plugin-threat-model.md)
- [ ] Disaster-recovery plan if the wallet file is lost (procedure to
  rotate to a new wallet, communicate the new public key)
- [ ] Backup of `ario/payload.json` artifacts independent of MLflow
  (so verification still works if MLflow's artifact store is lost)
