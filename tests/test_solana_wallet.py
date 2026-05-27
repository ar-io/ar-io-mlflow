"""Solana-default upload wallet support (upload/funding layer only).

The plugin defaults the upload/funding wallet to Solana (ed25519) while
keeping Arweave RSA working and selectable. Chain is detected from the
key's JSON shape — there is still ONE env var
(``ARIO_MLFLOW_ARWEAVE_WALLET``) and ONE path
(``DEFAULT_WALLET_PATH``). These tests pin that resolution.

No network (except the gated live-upload test). Mirrors the
``tmp_path`` + ``monkeypatch`` isolation style of the rest of the suite;
``DEFAULT_WALLET_PATH`` is monkeypatched per-test so nothing touches the
real ``~/.ario-mlflow/wallet.json``.
"""

from __future__ import annotations

import json
import os

import pytest

from ario_mlflow import arweave
from ario_mlflow.arweave import (
    WALLET_MODE_EPHEMERAL,
    WALLET_MODE_PERSISTENT,
    WALLET_MODE_USER,
    WALLET_TYPE_ARWEAVE,
    WALLET_TYPE_SOLANA,
    ArweaveAnchor,
    WalletLoadError,
)


@pytest.fixture(autouse=True)
def _isolate_wallet_env(tmp_path, monkeypatch):
    """No ambient wallet env, and the default path points into tmp.

    Each test that wants a *fresh install* uses the empty tmp path; tests
    that want a pre-existing wallet write to it first.
    """
    monkeypatch.delenv("ARIO_MLFLOW_ARWEAVE_WALLET", raising=False)
    monkeypatch.setattr(arweave, "DEFAULT_WALLET_PATH", str(tmp_path / "wallet.json"))


def _default_path() -> str:
    return arweave.DEFAULT_WALLET_PATH


# --- 1. Default is Solana (no network) -----------------------------------


def test_default_wallet_is_solana(tmp_path):
    """Fresh install, zero config → a Solana wallet is generated, persisted
    as a 64-int id.json, and Turbo routes with token=solana."""
    anchor = ArweaveAnchor()

    assert anchor.enabled is True
    assert anchor.wallet_type == WALLET_TYPE_SOLANA
    assert anchor.wallet_mode == WALLET_MODE_PERSISTENT
    assert anchor._token == "solana"

    # id.json written at the single path: a 64-int byte array.
    assert os.path.exists(_default_path())
    saved = json.loads(open(_default_path()).read())
    assert isinstance(saved, list) and len(saved) == 64
    assert all(isinstance(b, int) and 0 <= b <= 255 for b in saved)

    # Address is a base58 Solana pubkey (no network needed to derive it).
    addr = anchor._signer.get_wallet_address()
    assert isinstance(addr, str) and 32 <= len(addr) <= 44


# --- 2. Backward compat: pre-existing RSA wallet reused, never overwritten -


def test_existing_rsa_wallet_reused_and_not_overwritten():
    """A legacy RSA wallet.json is detected as arweave, reused unchanged,
    and the file is NOT overwritten (stable address across runs)."""
    rsa_jwk = ArweaveAnchor._generate_wallet()
    with open(_default_path(), "w") as f:
        json.dump(rsa_jwk, f)
    before = open(_default_path(), "rb").read()

    a1 = ArweaveAnchor()
    assert a1.enabled is True
    assert a1.wallet_type == WALLET_TYPE_ARWEAVE
    assert a1.wallet_mode == WALLET_MODE_PERSISTENT
    assert a1._token == "arweave"

    after = open(_default_path(), "rb").read()
    assert after == before, "existing wallet file must not be overwritten"

    # Address is stable across a second instantiation (no regeneration).
    a2 = ArweaveAnchor()
    assert a1._signer.get_wallet_address() == a2._signer.get_wallet_address()


# --- 3. Existing Solana wallet reused, stable address --------------------


def test_existing_solana_wallet_reused_stable_address():
    id_json = ArweaveAnchor._generate_solana_wallet()
    with open(_default_path(), "w") as f:
        json.dump(id_json, f)
    before = open(_default_path(), "rb").read()

    a1 = ArweaveAnchor()
    assert a1.wallet_type == WALLET_TYPE_SOLANA
    assert a1.wallet_mode == WALLET_MODE_PERSISTENT
    assert a1._token == "solana"
    assert open(_default_path(), "rb").read() == before  # not regenerated

    a2 = ArweaveAnchor()
    assert a1._signer.get_wallet_address() == a2._signer.get_wallet_address()


# --- 4. Explicit key via the single env var (user-configured) ------------


def test_explicit_rsa_jwk_via_env(tmp_path, monkeypatch):
    jwk = ArweaveAnchor._generate_wallet()
    p = tmp_path / "my-arweave.json"
    p.write_text(json.dumps(jwk))
    monkeypatch.setenv("ARIO_MLFLOW_ARWEAVE_WALLET", str(p))

    anchor = ArweaveAnchor()
    assert anchor.wallet_type == WALLET_TYPE_ARWEAVE
    assert anchor.wallet_mode == WALLET_MODE_USER
    assert anchor._token == "arweave"
    # The single default path is untouched when an explicit wallet is set.
    assert not os.path.exists(_default_path())


def test_explicit_solana_id_json_via_env(tmp_path, monkeypatch):
    id_json = ArweaveAnchor._generate_solana_wallet()
    p = tmp_path / "id.json"
    p.write_text(json.dumps(id_json))
    monkeypatch.setenv("ARIO_MLFLOW_ARWEAVE_WALLET", str(p))

    anchor = ArweaveAnchor()
    assert anchor.wallet_type == WALLET_TYPE_SOLANA
    assert anchor.wallet_mode == WALLET_MODE_USER
    assert anchor._token == "solana"


def test_explicit_solana_base58_via_env(tmp_path, monkeypatch):
    base58 = pytest.importorskip("base58")  # transitive dep of turbo-sdk
    id_json = ArweaveAnchor._generate_solana_wallet()
    b58 = base58.b58encode(bytes(id_json)).decode()
    p = tmp_path / "secret.json"
    p.write_text(json.dumps(b58))  # a JSON string
    monkeypatch.setenv("ARIO_MLFLOW_ARWEAVE_WALLET", str(p))

    anchor = ArweaveAnchor()
    assert anchor.wallet_type == WALLET_TYPE_SOLANA
    assert anchor.wallet_mode == WALLET_MODE_USER
    assert anchor._token == "solana"


# --- 5. Malformed user-configured key → WalletLoadError (both chains) -----


def test_malformed_incomplete_jwk_raises(tmp_path):
    p = tmp_path / "incomplete.json"
    p.write_text(json.dumps({"kty": "RSA", "n": "deadbeef"}))
    with pytest.raises(WalletLoadError, match="not a complete RSA JWK"):
        ArweaveAnchor._load_or_create_wallet(str(p))


def test_malformed_bad_base58_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps("not valid base58 !!!@@@"))
    with pytest.raises(WalletLoadError, match="not a valid Solana key"):
        ArweaveAnchor._load_or_create_wallet(str(p))


def test_malformed_wrong_length_array_raises(tmp_path):
    p = tmp_path / "shortarray.json"
    p.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(WalletLoadError, match="not a valid Solana id.json"):
        ArweaveAnchor._load_or_create_wallet(str(p))


def test_malformed_unrecognized_content_raises(tmp_path):
    p = tmp_path / "weird.json"
    p.write_text(json.dumps(42))  # neither object, array, nor string
    with pytest.raises(WalletLoadError, match="not a recognized format"):
        ArweaveAnchor._load_or_create_wallet(str(p))


def test_malformed_propagates_through_constructor(tmp_path, monkeypatch):
    """A malformed *caller-supplied* wallet raises out of __init__ — never
    a silent substitution (discipline preserved for both chains)."""
    p = tmp_path / "weird.json"
    p.write_text(json.dumps([1, 2, 3]))
    monkeypatch.setenv("ARIO_MLFLOW_ARWEAVE_WALLET", str(p))
    with pytest.raises(WalletLoadError):
        ArweaveAnchor()


# --- 6. Ephemeral fallback when persistence write fails -------------------


def test_ephemeral_fallback_when_write_fails(tmp_path, monkeypatch):
    """Fresh install but the path is unwritable (parent is a file, so
    makedirs raises OSError) → in-memory Solana wallet, mode=ephemeral,
    still enabled."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    monkeypatch.setattr(arweave, "DEFAULT_WALLET_PATH", str(blocker / "wallet.json"))

    anchor = ArweaveAnchor()
    assert anchor.enabled is True
    assert anchor.wallet_type == WALLET_TYPE_SOLANA
    assert anchor.wallet_mode == WALLET_MODE_EPHEMERAL
    assert anchor._token == "solana"


# --- 7. Never overwrite even a corrupt persistent wallet file -------------


def test_corrupt_persistent_wallet_not_overwritten():
    """An existing-but-unreadable wallet file is NEVER overwritten — the
    plugin degrades to an in-memory Solana wallet and leaves the file
    intact (no surprise data loss)."""
    with open(_default_path(), "w") as f:
        f.write("not json {{{")
    before = open(_default_path(), "rb").read()

    anchor = ArweaveAnchor()
    assert anchor.enabled is True
    assert anchor.wallet_type == WALLET_TYPE_SOLANA
    assert anchor.wallet_mode == WALLET_MODE_EPHEMERAL
    assert open(_default_path(), "rb").read() == before  # untouched


# --- 8. Gated live free-tier upload (opt-in; skipped by default / CI) -----


@pytest.mark.skipif(
    os.environ.get("ARIO_MLFLOW_LIVE_UPLOAD") != "1",
    reason="live Turbo upload gated behind ARIO_MLFLOW_LIVE_UPLOAD=1",
)
def test_live_solana_free_tier_upload():
    """Generate a fresh (zero-balance) Solana wallet and upload a small
    envelope to real Turbo on the free tier; assert a tx id returns and
    the bytes are retrievable from a gateway."""
    anchor = ArweaveAnchor()
    assert anchor.wallet_type == WALLET_TYPE_SOLANA

    envelope = {
        "spec_version": "ario.mlflow/v1",
        "event_id": "live-test-0001",
        "event_type": "training_complete",
        "subject": {"type": "mlflow_run", "run_id": "live-test"},
        "payload_hash": "0" * 64,
        "previous_hash": "GENESIS",
        "signed_at": "2026-01-01T00:00:00+00:00",
        "public_key": "00",
        "signature": "00",
    }
    result = anchor.upload_proof(envelope)
    assert result is not None, f"upload failed: {anchor.last_error}"
    tx_id = result["tx_id"]
    assert isinstance(tx_id, str) and len(tx_id) == 43

    fetched = anchor.fetch_proof(tx_id)
    assert fetched is not None
    assert fetched["event_id"] == "live-test-0001"
    print(f"\nLIVE Solana upload tx_id={tx_id} url={result['url']}")
