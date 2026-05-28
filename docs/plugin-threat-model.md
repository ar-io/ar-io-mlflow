# `ar-io-mlflow` threat model

What this plugin protects against, what it doesn't, and where the trust
boundaries actually sit. Required reading before a security review.

## TL;DR

`ar-io-mlflow` provides **tamper-evident provenance** for the MLflow lifecycle.
It commits cryptographic hashes of training/registration/inference events to
permanent public storage (Arweave via ar.io), so any party with the proof TX
can independently verify that what's in MLflow now matches what was anchored
at the time. The plugin does **not** prevent attacks; it makes them detectable
after the fact, by anyone, without trusting your infrastructure.

## Threats this defends against

### T1 — Model artifact swap after registration

**Scenario.** An attacker (or an honest mistake) replaces the bytes of a
registered model in MLflow's artifact store with a different model.

**Defense.** `VerifiedModel(...)` re-hashes the artifact at load time and
compares against `ario.artifact_hash` written at registration. A mismatch
raises `IntegrityError` *before* the underlying pyfunc model loads, so a
swapped model never executes user code. Production deployments should treat
`IntegrityError` as a security incident.

### T2 — MLflow tampering after anchoring

**Scenario.** A run's params, metrics, or trace tags are mutated in MLflow
after the proof was anchored. Goal: make MLflow appear to support a
different conclusion than what was actually anchored.

**Defense.** `verify_source_of_truth` re-derives the canonical bytes from a
*separate* live MLflow surface and compares to the anchored payload. If
either surface was modified after anchoring, the bytes won't agree and the
"Record Matches" row flips to FAIL.

### T3 — Anchored bytes tampering on Arweave

**Scenario.** An attacker substitutes a different envelope at the same TX,
or proxies fetches through a malicious gateway returning forged bytes.

**Defense.** Arweave is content-addressed and immutable; rewriting the
bytes at a TX isn't possible once finalized. Multi-gateway fetch (default
`turbo-gateway.com` + `ardrive.net`) means a single malicious gateway
can't lie undetected — fetches that don't match the SHA-256 commitment in
the envelope fail the `verify_anchored_bytes` check.

### T4 — Loss of MLflow tracking server

**Scenario.** Your MLflow server is decommissioned, migrated, or
bit-rotted. Three years later an auditor asks for proof a model existed
at time T.

**Defense.** The proof envelope and canonical payload (in MLflow's
artifact store) survive independently of the tracking server. The
envelope on Arweave is permanent. As long as the artifact store (S3, GCS,
local FS) survives, verification still works against any RFC-8785
implementation in any language — no dependency on this plugin or on a
running MLflow server.

### T5 — Lying about training

**Scenario.** Someone claims "this model was trained at time T with these
params." Without proof, the claim is unverifiable.

**Defense.** The signed envelope binds `event_id`, `subject` (run ID),
`payload_hash`, `previous_hash`, `signed_at`, and `public_key` together
with an Ed25519 signature. The signature requires possession of the
private key at signing time. Anyone with the embedded `public_key` can
verify the claim independently — there's no "trust us" step.

### T6 — Silent identity swap during signing

**Scenario.** An operator sets `ARIO_MLFLOW_ARWEAVE_WALLET=/path/to/key`
but the file is missing or malformed at runtime. A naive plugin would
silently auto-generate a different wallet and continue signing — proofs
would land on-chain under a different address than the operator intended,
with no programmatic signal.

**Defense.** `ArweaveAnchor` raises `WalletLoadError` from the constructor
when a caller-supplied wallet path is unloadable. The plugin refuses to
substitute an auto-generated wallet for an operator-named one.

### T7 — Independent third-party verification

**Scenario.** A regulator wants to verify a proof without trusting your
infrastructure or the plugin's correctness.

**Defense.** ar.io Verify (an independent gateway-operator-run service)
re-fetches the bytes, recomputes the hash against the gateway's own digest,
and verifies the Ed25519 signature against the embedded public key. The
verification result is itself signed by the operator. Regulators can
verify the operator's signature against their public key with standard
RSA-PSS SHA-256 — entirely outside this plugin, in any RFC-8785-capable
environment.

## Threats this does NOT defend against

### N1 — CI runner / training environment compromise *before* anchoring

**Scenario.** An attacker controls the training environment, runs a
modified training script, anchors the result. The signed envelope
attests faithfully to what was trained — but what was trained is
attacker-controlled.

**Why not defended.** Cryptographic anchoring binds a claim to an
identity; it doesn't validate the claim's underlying correctness. Use
attested compute (TEE, Nitro Enclaves, sigstore-style provenance for
the training image) for this layer. Out of scope for this plugin.

### N2 — Compromised signing key

**Scenario.** An attacker steals the private key (`~/.ario-mlflow/keys/`
or your secrets-manager copy) and signs forgeries.

**Why not defended.** Standard cryptography assumption: a stolen private
key produces valid signatures. Mitigations are key hygiene
(`ARIO_MLFLOW_SIGNING_KEY` from a secrets manager, regular rotation,
isolating signing to a hardened service) and detection (chain anomalies,
unexpected `previous_hash` values, signing volume alerts).

### N3 — Insider with wallet access

**Scenario.** Same as N2 but legitimate access. Someone with the wallet
anchors something they shouldn't.

**Why not defended.** Authorization, not authentication, is the issue.
Use organizational controls (separation of duties, dual-signed
deployments, immutable training pipelines) — the plugin records who
signed what, not who *should* have.

### N4 — Semantic correctness

**Scenario.** A model is anchored honestly, but the model itself is bad
(biased, wrong objective, hallucinating in production).

**Why not defended.** Provenance proves *what was committed*, not *that
it was right*. Semantic verification (whether *this model* produced
*this output* for *this input* correctly) is a separate problem the
plugin intentionally doesn't solve — out of scope by design, not a
pending feature.

### N5 — Fake compliance via "anchored garbage"

**Scenario.** A bad actor anchors plausible-looking but fake training
metadata, points at it as "proof of compliance."

**Why not defended.** The plugin signs whatever you give it. Verification
proves the signature's identity and that the bytes haven't changed —
not that the original claim was reasonable. Compliance auditors must
still inspect *what* was anchored, not just *that* something was anchored.

### N6 — Training data leakage

**Scenario.** Sensitive training data ends up on Arweave permanently.

**Why not defended *and* explicitly out of scope.** The plugin commits to
hashes of params, metrics, and artifact checksums — not raw data. Source
data stays in MLflow. The standalone-dataset envelope commits to
`name`, `source` URI, `digest`, and `schema_hash` — not row contents.
**Callers passing PII or business data via `metadata={...}` are
responsible for not doing so**; the plugin doesn't redact arbitrary
caller-supplied fields.

## Trust boundaries

For verification to be meaningful, an auditor trusts:

1. **The Ed25519 algorithm and the JCS canonicalization scheme** — both
   are open standards (RFC-8032, RFC-8785) implementable in any
   language; no proprietary cryptography.
2. **At least one ar.io gateway** — multi-gateway fetch defends against
   single-gateway compromise; independent re-verification with a
   different gateway provides defense-in-depth.
3. **The signer's public key bound to a known identity** — out of scope
   for the plugin. Production deployments should publish their public
   key via a trusted channel (signed announcement, web PKI'd domain,
   organization key transparency log) so verifiers can confirm
   "envelope signed by `<known org>`" not just "envelope signed by
   *some* key."
4. **Arweave's permanence assumption** — the network has held data
   continuously since 2018; the failure mode is "all gateways serving
   the same TX go down" not "the bytes change."

The plugin **does not** require trusting:

- The MLflow server (its data is the source of truth, but tampering is
  detectable post-anchor)
- Any single ar.io gateway operator
- This plugin's correctness — verification is implementable from the
  envelope spec alone, no `ar-io-mlflow` install needed
