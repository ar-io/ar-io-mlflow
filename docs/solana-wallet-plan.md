# Solana wallet support — implementation plan (historical, shipped in 0.2.0)

> **Status: HISTORICAL — fully shipped in 0.2.0 (2026-05-27).** This document was the design plan that drove the implementation; it's kept for reference but the live behavior is what the code, `README.md`, and CHANGELOG `[0.2.0]` describe today. New work shouldn't be specced against this doc — check `ario_mlflow/arweave.py` and the CHANGELOG first. The original plan text follows below.

> **Original goal:** Make the upload wallet default to **Solana** while keeping **Arweave RSA** fully working and user-selectable. Confined to the upload/funding layer — the Ed25519 *signing* key (`ProofEngine`, the envelope trust anchor) is unchanged, and so is the envelope/proof format. The verifier (`verify_record`) and cross-conformance with `ar-io-agent` are untouched.

## 0. Why / what changes

`ario_mlflow/arweave.py` (`ArweaveAnchor`) signs ANS-104 data items and uploads them to Arweave via Turbo. Today it always uses an **Arweave RSA-4096 JWK** wallet. We want **Solana (ed25519) as the default funding wallet**, with RSA still available. The destination is still Arweave (via Turbo); only the *signer/funding chain* changes — so the `ArweaveAnchor` name and the "anchored to Arweave" framing remain accurate. This is unblocked by **`turbo-sdk>=0.1.0`** which ships `SolanaSigner` (ANS-104 signature type 2). See `ar-io-agent/docs/evidence-bundle.md`-adjacent context and the upstream PRs (`ardriveapp/turbo-python-sdk#7`, `#8`).

**Decisions (locked):**
- **Solana is the default** for newly generated wallets.
- **RSA keeps working**, and the user can explicitly pick either chain.
- **Backward compatible:** an already-onboarded user with a persistent RSA wallet on disk keeps using it — no forced migration.
- **Two orthogonal axes:** `wallet_type ∈ {solana, arweave}` (NEW) and the existing `wallet_mode ∈ {user-configured, persistent, ephemeral}` (persistence — unchanged in meaning).

## 1. Current state (verified)

- `__init__` (arweave.py:98) loads a wallet, constructs `ArweaveSigner(jwk)` → `Turbo(signer)` → caches `self._upload_url`, `self._token`, sets `self.wallet_mode`.
- `_load_or_create_wallet` (arweave.py:251): user-configured (`ARIO_MLFLOW_ARWEAVE_WALLET` / `wallet_path`) → persistent (`DEFAULT_WALLET_PATH = ~/.ario-mlflow/wallet.json`) → ephemeral. Validates `_REQUIRED_JWK_FIELDS`. Raises `WalletLoadError` on a malformed *caller-supplied* wallet (never silently substitutes).
- `_generate_wallet` (arweave.py:327): RSA-4096 → JWK dict.
- `upload_proof` (arweave.py:352): `create_data` + `sign` from `turbo_sdk.bundle`, POST `{self._upload_url}/tx/{self._token}`. **Signer-agnostic** — already routes by `self._token`, so it works for Solana with no change.
- Dep pin: `pyproject.toml` `turbo-sdk>=0.0.5`.
- `wallet_mode` is surfaced in logs (arweave.py:160-172), in `report.py`'s wallet-mode banner, and in CLI/status output.

## 2. Design — one path, one env var, type detected by key shape

Guiding principle: **wallet `type` is a property of the key you have, not a knob to configure.** An RSA JWK is a JSON *object* (`kty/n/e/d…`); a Solana key is a JSON *array of 64 ints* (CLI `id.json`) or a base58 *string* — unambiguous JSON types, so detect the chain by shape. This keeps the surface identical to today: **no new env vars, no new wallet paths.**

### 2.1 Wallet resolution (generalize `_load_or_create_wallet`, keep `WalletLoadError`)

Keep the single env var `ARIO_MLFLOW_ARWEAVE_WALLET` and the single path `DEFAULT_WALLET_PATH = ~/.ario-mlflow/wallet.json`. Both now accept **either** chain's key; type is detected from content.

1. **User-configured** (`ARIO_MLFLOW_ARWEAVE_WALLET` / `wallet_path` set) → load the file, detect shape, build the matching signer. Malformed-for-its-shape → `WalletLoadError` (never silently substitute). `mode=user-configured`.
2. **Persistent reuse:** if `DEFAULT_WALLET_PATH` exists → load + detect shape + use it. **A legacy RSA `wallet.json` is detected as Arweave and reused as-is** (backward compatible for free). **Never overwrite an existing wallet file** — so no surprise address change, no data loss. `mode=persistent`.
3. **Fresh install (no wallet present):** **generate a Solana keypair** (the new default), persist it as a Solana CLI `id.json` (64-int array) to `DEFAULT_WALLET_PATH`, `mode=persistent`. If the write fails → in-memory, `mode=ephemeral`.

Return `(secret_material, wallet_type, mode)` where `wallet_type ∈ {solana, arweave}` is the *detected* chain.

**Shape detection** (single helper):
- JSON `dict` with the RSA JWK fields → `arweave` (validate via existing `_REQUIRED_JWK_FIELDS`).
- JSON `list` of 64 ints, or a base58 `str` → `solana` (validate by attempting `SolanaSigner(...)`).
- Anything else → `WalletLoadError`.

### 2.2 Env vars / paths — UNCHANGED

No new env vars, no new paths. `ARIO_MLFLOW_ARWEAVE_WALLET` now accepts an RSA JWK *or* a Solana key (auto-detected); to use RSA, point it at your JWK — to use a specific Solana key, point it at your `id.json`/base58. The name is legacy-awkward (it predates Solana); leave it for backward compat and note a generic-rename (`ARIO_MLFLOW_WALLET`, with the old name as alias) as an optional future nicety — not part of this change.

### 2.3 Signer construction (`__init__`)

Branch on the detected `wallet_type`:
- `arweave` → `ArweaveSigner(jwk)` (as today).
- `solana` → `SolanaSigner(secret)` (from `turbo_sdk`).

Then `Turbo(signer)` as today — `turbo.token` auto-resolves to `"arweave"` / `"solana"`, and `upload_proof`'s `/tx/{self._token}` routes correctly with **no change**. `signer.get_wallet_address()` works for both (Solana → base58). Set a derived `self.wallet_type` attribute.

### 2.4 Solana key generation + format

Use `cryptography` (already a dep), no new dependency:
```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
priv = Ed25519PrivateKey.generate()
seed = priv.private_bytes_raw()                 # 32 bytes
pub  = priv.public_key().public_bytes_raw()     # 32 bytes
id_json = list(seed + pub)                       # Solana CLI id.json (64 ints)
```
Persist `id_json` as JSON at `DEFAULT_WALLET_PATH` (`~/.ario-mlflow/wallet.json` — the single existing path; a Solana `id.json` array and an RSA JWK object are distinguishable by shape on load), `chmod 0o600`. Only written when no wallet file already exists (never overwrite). `SolanaSigner` accepts the 64-int list directly.

### 2.5 Reporting / logging / status

- Add `wallet_type` to the enable/log lines (arweave.py:160-172) — e.g. `Arweave anchoring enabled (chain=solana, wallet=<addr>, mode=persistent)`.
- `report.py` wallet banner: include the chain ("Signed with a Solana wallet …" / "… Arweave wallet …") alongside the existing persistent/ephemeral transparency text. Keep it light.
- Any `status`/CLI surface that prints the wallet should show `wallet_type`.

### 2.6 Explicitly NOT changing

- **No new on-chain Arweave tags** — keep the conservative tag policy in `_build_default_tags`. (The owner address already encodes the chain.)
- **No rename** of `ArweaveAnchor` or `ARIO_MLFLOW_ARWEAVE_WALLET` (backward compat; the anchor target is still Arweave). Note as a possible future cosmetic cleanup only.
- **No change** to `ProofEngine` / the Ed25519 envelope signing key, the envelope/proof format, `verify_record`, or cross-conformance with `ar-io-agent`.

## 3. Dependency

`pyproject.toml`: `turbo-sdk>=0.0.5` → **`turbo-sdk>=0.1.0`** (SolanaSigner + `TOKEN_MAP[2]="solana"` landed in 0.1.0, now on PyPI).

## 4. Test plan (`tests/`, mirror `test_input_anchoring.py` style: `tmp_path` + `monkeypatch`)

Keep `DEFAULT_WALLET_PATH` monkeypatchable (existing module constant). Cover:

1. **Default is Solana:** fresh env (no env var, path → empty tmp) → `wallet_type=="solana"`, a Solana `id.json` created at the path, `Turbo(...).token=="solana"`, address is base58. (No network.)
2. **Backward-compat:** pre-existing RSA `wallet.json` at the path, no env → `wallet_type=="arweave"`, reused, **and the file is not overwritten** (assert bytes/address unchanged after instantiation).
3. **Existing Solana wallet reused:** pre-existing Solana `id.json` at the path → `solana`, reused, stable address (no regeneration).
4. **Explicit key via the env var** (`ARIO_MLFLOW_ARWEAVE_WALLET`): RSA JWK → `arweave`; Solana `id.json` → `solana`; base58 Solana secret → `solana`. All `user-configured`. (RSA case keeps existing tests green.)
5. **Malformed key** → `WalletLoadError`: RSA object missing JWK fields (existing), bad base58, wrong-length array, and unrecognizable content.
6. **Ephemeral fallback:** fresh install but persistence write fails (monkeypatch `os.makedirs`/`open` to raise) → Solana `mode=="ephemeral"`, still enabled.
7. **Live free-tier upload (gated, opt-in):** behind `ARIO_MLFLOW_LIVE_UPLOAD=1` (skip by default and in CI). Generate a Solana wallet, `upload_proof` a <100 KiB envelope to real Turbo, assert a tx id returns and the bytes are retrievable from a gateway. Mirrors the SDK-layer proof (a fresh zero-balance Solana key works on free-tier).

Run the repo's existing lint/format/type/test gates (check `pyproject.toml` / CI) and keep everything green; don't regress existing anchor tests.

## 5. Acceptance criteria

- Fresh install → Solana wallet auto-generated; `upload_proof` succeeds with `token=solana`. Zero config required.
- Existing RSA `wallet.json` → detected and used unchanged (`arweave`), never overwritten; uploads succeed.
- User can bring either chain's key by pointing `ARIO_MLFLOW_ARWEAVE_WALLET` at it (type auto-detected).
- `WalletLoadError` discipline preserved for both chains (no silent substitution).
- `turbo-sdk>=0.1.0`; full test suite green; new tests added; the gated live Solana upload passes when enabled.
- Logs/report/status show the wallet chain.

## 6. Out of scope (follow-ups)

- `ar-io-agent` Go sigType-2 Solana path (separate workstream; arbundles pins the exact spec: type 2, raw deep hash, ed25519, 32-byte owner, base58 address).
- Renaming `ArweaveAnchor` / the env var.
- Funding/top-up flows (free-tier <100 KiB covers the plugin's small envelopes).
