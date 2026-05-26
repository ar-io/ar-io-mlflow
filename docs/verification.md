# Verification

This is the deep reference for verifying `ar-io-mlflow` proofs — from the
command line, from your own Python, and from any language with no plugin
installed. For the quickstart and the CLI cheat-sheet, see the
[README](../README.md#cli); for the design rationale, see
[`architecture.md`](architecture.md).

## The four checks

Every verification surface — the CLI, the library functions, and the
language-neutral auditor recipe — composes the same four checks. Each one
answers a different question, and they fail independently so you can see
exactly what broke.

| # | Check | Question it answers | Needs |
|---|-------|---------------------|-------|
| 1 | **Signature** | Did the holder of the embedded key sign these exact bytes, under a spec version we understand? | nothing (offline) |
| 2 | **Anchored bytes** | Does `ario/payload.json` in MLflow still hash to the envelope's `payload_hash`? | MLflow access |
| 3 | **Source of truth** | Do the *live* MLflow fields (params/metrics, or the prediction's trace tag) still re-derive the anchored payload? | MLflow access |
| 4 | **ar.io attestation** *(optional)* | Has an independent ar.io gateway operator confirmed the TX is permanently stored? | `ARIO_MLFLOW_ARIO_VERIFY_URL` |

Checks 2 and 3 are what bind the proof to **your MLflow** — they catch
post-anchor tampering of the artifact (check 2) and of the live tracking
data (check 3). Check 1 only proves the envelope is internally
self-consistent; check 4 is the only one that independently confirms the
on-chain copy. See [`plugin-threat-model.md`](plugin-threat-model.md) for
what this does and does not defend against.

### Three-valued results: `ok` is `True`, `False`, or `None`

Every check returns a dict with an `ok` field that is **tri-state**, and
misreading `None` as failure (or success) is the most common mistake:

- `ok=True` — the check ran and passed.
- `ok=False` — the check ran and **failed**. Something is wrong.
- `ok=None` — the check **did not run / is not applicable** (e.g. no
  MLflow client was supplied, or this event type has no payload artifact).
  Not a pass and not a fail.

Most check dicts also carry a `reason` string when `ok` is `False` or
`None` (e.g. `"no_ario_client"`, `"unsupported_spec_version"`,
`"live_refetch_incomplete"`), so you can tell *why* without guessing.

## Which function do I call?

There are three composite entry points, all re-exported from the top-level
`ario_mlflow` package. Pick based on **what you have in hand** and **whether
you have MLflow access**:

| Function | Use when | MLflow needed? | Runs check 3? |
|----------|----------|:---:|:---:|
| `verify_record(envelope, canonical_bytes, …)` | You're an **auditor** holding a portable bundle (the envelope + the canonical bytes). No operator infra. | no | no |
| `verify_proof_by_tx(tx_id, …)` | You're the **operator** and only have a TX ID — fetch the envelope from Arweave and run everything. | optional | yes (if client given) |
| `full_verify(envelope, …)` | You're the **operator** and already hold the envelope (e.g. from a tag). | optional | yes (if client given) |

The single-check functions (`verify_signature`, `verify_anchored_bytes`,
`verify_source_of_truth`, `verify_ario_attestation`) are also public if you
want to run just one.

### Constructing the components

The composite functions take collaborators as keyword args. Build them the
same way the CLI does:

```python
import os
from ario_mlflow.proof import ProofEngine
from ario_mlflow.arweave import ArweaveAnchor
from ario_mlflow.verify import ArioVerifyClient

proof_engine = ProofEngine()                       # loads/derives the signing identity
anchor = ArweaveAnchor(                             # fetches envelopes from Arweave
    os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", ""),
    os.environ.get("ARIO_MLFLOW_GATEWAY_HOST", "turbo-gateway.com"),
)
ario_client = ArioVerifyClient()                   # optional; check 4. Disabled unless
                                                   # ARIO_MLFLOW_ARIO_VERIFY_URL is set.
```

> `ProofEngine` and `ArweaveAnchor` are **not** top-level exports — import
> them from `ario_mlflow.proof` and `ario_mlflow.arweave`. The verify
> functions themselves are top-level (`from ario_mlflow import
> verify_proof_by_tx`).

## Examples

### Operator: verify by TX ID (the common case)

You have a TX ID from an MLflow tag (`ario.training_tx`, etc.) and want the
full picture:

```python
from mlflow.tracking import MlflowClient
from ario_mlflow import verify_proof_by_tx

result = verify_proof_by_tx(
    tx_id,
    anchor=anchor,
    proof_engine=proof_engine,
    mlflow_client=MlflowClient(),      # enables checks 2 and 3
    ario_client=ario_client,           # enables check 4 (optional)
)

if not result["proof_found"]:
    raise SystemExit("envelope not retrievable from Arweave")
if result["overall"] is not True:
    raise SystemExit(f"verification failed: {result}")
```

`result` shape:

```python
{
  "proof_found": True,
  "signature":       {"ok": True, "signature_valid": True,
                      "spec_version_status": "supported", "legacy_envelope": False},
  "anchored_bytes":  {"ok": True, "computed_hash": "…", "stored_hash": "…",
                      "payload_bytes": b"…", "artifact_expected": True},
  "source_of_truth": {"ok": True, "rebuilt_bytes": b"…",
                      "live_fields_refetched": ["artifact_checksums", "git_commit",
                                                "metrics", "params", "source_name"]},
  "ario_attestation":{"ok": None, "reason": "no_ario_client"},   # or level details
  "overall": True,
}
```

### Auditor: verify a portable bundle offline

No MLflow, no operator infra — just the envelope and the canonical bytes it
committed to (e.g. shipped together in an evidence bundle):

```python
from ario_mlflow import verify_record

result = verify_record(envelope, canonical_bytes, proof_engine=proof_engine)
# Optionally pass ario_client=… to add the independent on-chain attestation.

assert result["overall"] is True
```

`verify_record` runs checks 1, 2 (against the bytes you supply), and
optional 4 — **it deliberately omits check 3**, because for an auditor the
bundle *is* the source of truth; there's no live MLflow to re-derive from.
Result shape:

```python
{
  "signature":       {"ok": True, "signature_valid": True,
                      "spec_version_status": "supported", "legacy_envelope": False},
  "anchored_bytes":  {"ok": True, "computed_hash": "…", "stored_hash": "…"},
  "ario_attestation":{"ok": None, "reason": "no_ario_client"},
  "overall": True,
}
```

### Gotcha: `full_verify` / `verify_proof_by_tx` without an MLflow client

For training, registration, and prediction events, checks 2 and 3 are
**required** — a `None` (not just a `False`) on either makes `overall`
fail. So calling the operator functions without an `mlflow_client` returns
`overall=False` even when the signature is perfectly valid:

```python
from ario_mlflow import full_verify

full_verify(env, proof_engine=proof_engine)["overall"]
# -> False   (anchored_bytes.ok == None -> required check didn't run)

full_verify(env, proof_engine=proof_engine, mlflow_client=MlflowClient())["overall"]
# -> True
```

This is intentional: for these event types, "I couldn't check MLflow" is
not allowed to read as a green light. If you genuinely want an
offline/signature-only verdict, use `verify_record` (auditor semantics) or
read `result["signature"]["ok"]` directly.

## `spec_version` and cross-product verification

Every envelope this plugin mints carries
`spec_version: "ario.mlflow/v1"` in its signed body (exported as
`ario_mlflow.SPEC_VERSION`). The signature check classifies it:

- **supported** — `ario.mlflow/v1` *or* `ario.agent/v1`. Verification
  proceeds normally.
- **legacy** — field absent. Envelopes anchored before this field existed
  still verify; the result carries `legacy_envelope: True` so you can tell.
- **unsupported** — an unknown major (e.g. `ario.mlflow/v99`). The
  signature result returns `ok=False, reason="unsupported_spec_version"`
  and `overall` fails, even if the bytes are otherwise validly signed.

Because `ario.agent/v1` is in the accepted set, **this plugin verifies
envelopes minted by the sister [`ar-io-agent`](https://github.com/ar-io/ar-io-agent)
daemon**, and vice versa — the two share the envelope spec and crypto. The
agent's CI runs `verify_record` against agent-produced envelopes to prove
it; `tests/test_plugin_smoke.py::test_verify_commitment_accepts_cross_product_agent_envelope`
proves the reverse.

## Verifying in CI / monitoring

[`plugin-production.md`](plugin-production.md#monitoring-and-alerting) covers
*what to alert on*; this is the verify call to put in that job. A scheduled
re-verification that fails loudly when a previously-anchored model stops
verifying:

```python
import sys
from mlflow.tracking import MlflowClient
from ario_mlflow import verify_proof_by_tx
from ario_mlflow.proof import ProofEngine
from ario_mlflow.arweave import ArweaveAnchor

client = MlflowClient()
proof_engine, anchor = ProofEngine(), ArweaveAnchor("", "turbo-gateway.com")

failures = []
for mv in client.search_model_versions("name='fraud-detector'"):
    tx = mv.tags.get("ario.registration_tx")
    if not tx:
        continue
    result = verify_proof_by_tx(tx, anchor=anchor, proof_engine=proof_engine,
                                mlflow_client=client)
    if result["overall"] is not True:
        failures.append((mv.version, result))

if failures:
    for version, result in failures:
        print(f"FAIL v{version}: {result['overall']!r}", file=sys.stderr)
    sys.exit(1)
```

Run it on a cron / CI schedule and page on a non-zero exit. Because the
proof lives on permanent storage, this keeps working even if the original
training environment is long gone — that's the auditor-independence
property. The one expected non-tampering failure mode is a pruned
prediction trace, which surfaces as
`source_of_truth.reason == "live_refetch_incomplete"` rather than a silent
pass; the signature + anchored-bytes + ar.io layers remain verifiable.

## The HTML report (`ario/verification.html`)

On every anchored event the plugin writes an `ario/verification.html`
artifact, viewable directly in the **MLflow artifact viewer** for the run
(open the run → Artifacts → `ario/` → `verification.html`). It's the
human-facing companion to the machine checks and renders:

- A status badge — **Signed (local)** → **Anchored** → **Verified
  (Level N)** as the proof matures.
- The envelope details: `payload_hash`, `previous_hash`, signature, public
  key, Arweave TX ID and gateway URL.
- Artifact-integrity status.
- A **copy-pasteable CLI verify command** so any reader can re-check the
  proof themselves.
- A **wallet-mode transparency notice** — flags when the proof was signed
  with the plugin's auto-generated demo wallet rather than a
  caller-configured production wallet, so nobody mistakes a demo signature
  for a production one.

The CLI `verify` commands regenerate this report (with the latest
verification result and ar.io attestation) and write the outcome back to
the run's `ario.*` tags.

## Verifying without Python (any language)

The proof spec is language-neutral — see the
[auditor recipe in the README](../README.md#verifying-without-python):
fetch the envelope, JCS-canonicalize the body minus `signature`, Ed25519-
verify against the embedded `public_key`, re-hash `ario/payload.json`
against `payload_hash`, and (optionally) walk the `previous_hash` chain. Add
one step today: read `spec_version` and accept the `ario.mlflow/v*` /
`ario.agent/v*` majors you understand, rejecting unknown ones.

## See also

- [README — CLI](../README.md#cli) and
  [Verifying without Python](../README.md#verifying-without-python)
- [`architecture.md`](architecture.md) — pure-commitment design, per-event
  chains, JCS canonicalization
- [`plugin-threat-model.md`](plugin-threat-model.md) — trust boundaries;
  what the checks do and don't prove
- [`plugin-production.md`](plugin-production.md) — wallet ops, CI/CD,
  monitoring, runbooks
