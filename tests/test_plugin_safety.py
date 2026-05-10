"""Tests for the plugin's safety pass: loud failure on caller-intent
violations, no developer-URL fallback in shipped reports, version sync.

These tests pin the behavior introduced when ``ario-mlflow`` was hardened
for formal ar.io product status. The headline guarantee: if an operator
explicitly names a wallet (via ``ARIO_MLFLOW_ARWEAVE_WALLET`` or the
``wallet_path`` argument), the plugin **must not** silently sign with a
different identity. Failures must be loud.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ario_mlflow import arweave
from ario_mlflow.arweave import (
    WALLET_MODE_PERSISTENT,
    WALLET_MODE_USER,
    ArweaveAnchor,
    WalletLoadError,
)
from ario_mlflow.report import generate_verification_html


# --- Wallet load: caller-intent violations must raise --------------------


def test_load_wallet_raises_when_caller_path_missing(tmp_path):
    """Operator named a wallet path that doesn't exist. Refuse to silently
    auto-generate — the operator's intent must surface as an error."""
    missing = tmp_path / "definitely-not-here.json"
    with pytest.raises(WalletLoadError, match="does not exist"):
        ArweaveAnchor._load_or_create_wallet(str(missing))


def test_load_wallet_raises_when_file_is_not_json(tmp_path):
    """File exists at the supplied path but is not valid JSON. Loud
    failure — without it, proofs would land on-chain under a different
    auto-generated identity with no programmatic signal."""
    bad = tmp_path / "garbage.json"
    bad.write_text("this is not json {{{")
    with pytest.raises(WalletLoadError, match="not valid JSON"):
        ArweaveAnchor._load_or_create_wallet(str(bad))


def test_load_wallet_raises_when_jwk_is_incomplete(tmp_path):
    """File parses as JSON but is missing required RSA JWK fields. Same
    failure category as malformed: caller's intent was to use this
    wallet, and the wallet is unusable."""
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"kty": "RSA", "n": "deadbeef"}))
    with pytest.raises(WalletLoadError, match="not a complete RSA JWK"):
        ArweaveAnchor._load_or_create_wallet(str(incomplete))


def test_load_wallet_accepts_valid_jwk(tmp_path):
    """Happy path — well-formed RSA JWK loads as ``user-configured``."""
    required = ("kty", "n", "e", "d", "p", "q", "dp", "dq", "qi")
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({f: "x" for f in required}))
    jwk, mode = ArweaveAnchor._load_or_create_wallet(str(valid))
    assert mode == WALLET_MODE_USER
    assert set(required).issubset(jwk)


def test_load_wallet_falls_back_to_default_when_no_caller_path(tmp_path, monkeypatch):
    """No ``wallet_path`` supplied → auto-generate at the default path.
    Caller's intent is silent here (they opted out of explicit
    configuration), so generating a wallet is the right default."""
    fake_default = tmp_path / "auto-wallet.json"
    monkeypatch.setattr(arweave, "DEFAULT_WALLET_PATH", str(fake_default))
    jwk, mode = ArweaveAnchor._load_or_create_wallet("")
    assert mode == WALLET_MODE_PERSISTENT
    assert fake_default.exists()
    assert {"kty", "n", "e", "d"}.issubset(jwk)


# --- Public API: WalletLoadError is part of the package surface ----------


def test_walletloaderror_is_re_exported_from_package():
    """Callers should be able to ``from ario_mlflow import WalletLoadError``
    to handle wallet-load failures without reaching into a private
    submodule."""
    import ario_mlflow

    assert ario_mlflow.WalletLoadError is WalletLoadError
    assert "WalletLoadError" in ario_mlflow.__all__


# --- Report rendering: no developer-URL fallback in shipped output -------


def _minimal_proof():
    return {
        "event_type": "training_complete",
        "subject": {"type": "mlflow_run", "run_id": "r-1"},
        "payload_hash": "0" * 64,
        "previous_hash": "GENESIS",
        "signed_at": "2026-01-01T00:00:00+00:00",
        "public_key": "ab" * 32,
        "signature": "cd" * 64,
        "record": {"event_type": "training_complete", "run_id": "r-1", "timestamp": "2026-01-01"},
    }


def _minimal_anchor():
    return {"tx_id": "TX-abc", "url": "https://turbo-gateway.com/TX-abc", "receipt": {}}


def test_report_does_not_render_developer_fallback_url(monkeypatch):
    """When neither caller arg nor env var sets the verify URL, the
    rendered HTML must not contain the historical ``vilenarios.com``
    developer endpoint. Shipping that URL by default leaks a personal
    domain into every artifact."""
    monkeypatch.delenv("ARIO_MLFLOW_ARIO_VERIFY_URL", raising=False)

    html_out = generate_verification_html(
        proof=_minimal_proof(),
        anchor_result=_minimal_anchor(),
    )

    assert "vilenarios.com" not in html_out
    # CLI command should still appear — that part is always actionable.
    assert "ario-mlflow verify run" in html_out


def test_report_omits_external_link_when_no_verify_url_configured(monkeypatch):
    """No verify URL configured → no ``check manually on ar.io Verify``
    link. The CLI command stands alone."""
    monkeypatch.delenv("ARIO_MLFLOW_ARIO_VERIFY_URL", raising=False)

    html_out = generate_verification_html(
        proof=_minimal_proof(),
        anchor_result=_minimal_anchor(),
    )

    assert "check manually on ar.io Verify" not in html_out


def test_report_includes_external_link_when_verify_url_set(monkeypatch):
    """Verify URL configured (caller arg) → external link IS rendered,
    pointing at the configured base + tx_id."""
    monkeypatch.delenv("ARIO_MLFLOW_ARIO_VERIFY_URL", raising=False)

    html_out = generate_verification_html(
        proof=_minimal_proof(),
        anchor_result=_minimal_anchor(),
        verify_base_url="https://verify.example.com/check",
    )

    assert "https://verify.example.com/check/TX-abc" in html_out
    assert "check manually on ar.io Verify" in html_out


def test_report_uses_env_verify_url_when_no_caller_arg(monkeypatch):
    """Verify URL configured via env var → external link IS rendered."""
    monkeypatch.setenv("ARIO_MLFLOW_ARIO_VERIFY_URL", "https://env.example.com")

    html_out = generate_verification_html(
        proof=_minimal_proof(),
        anchor_result=_minimal_anchor(),
    )

    assert "https://env.example.com/TX-abc" in html_out


# --- Version sync: __version__ matches pyproject.toml --------------------


def test_version_is_importable_from_package():
    """``ario_mlflow.__version__`` is part of the public API for callers
    that need to log or report which plugin version produced a proof."""
    from ario_mlflow import __version__

    assert isinstance(__version__, str)
    assert __version__  # non-empty


def test_version_matches_pyproject_toml():
    """The runtime ``__version__`` and the packaging metadata in
    ``pyproject.toml`` must agree — otherwise PyPI's reported version
    and the runtime's reported version drift."""
    from ario_mlflow import __version__

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    text = pyproject.read_text()
    assert f'version = "{__version__}"' in text, (
        f"pyproject.toml does not declare version = \"{__version__}\""
    )
