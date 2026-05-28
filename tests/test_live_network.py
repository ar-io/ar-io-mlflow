"""Live-network smoke tests — the dev-shipping gate.

These exercise the only paths the rest of the suite can't: real Arweave
uploads via Turbo, real gateway fetches, and (optionally) real ar.io Verify
attestation. They're slow (network + Arweave indexing propagation),
they consume real Turbo credits, and they require a **funded** wallet —
so they're opt-in.

To run:

    ARIO_MLFLOW_LIVE_NETWORK=1 pytest tests/test_live_network.py -v -s

Wallet resolution follows the production rules (``ARIO_MLFLOW_ARWEAVE_WALLET``
env var → ``~/.ario-mlflow/wallet.json`` → auto-generate Solana).
An auto-generated wallet is unfunded, so the upload+fetch test will detect
"insufficient credits" via ``anchor.last_error`` and skip cleanly rather
than fail confusingly.

The ar.io attestation test is gated separately on
``ARIO_MLFLOW_ARIO_VERIFY_URL`` being set — skipped otherwise.
"""

from __future__ import annotations

import os
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ARIO_MLFLOW_LIVE_NETWORK") != "1",
    reason="live-network smoke test; set ARIO_MLFLOW_LIVE_NETWORK=1 to run",
)


# Arweave indexing propagation: a freshly uploaded tx isn't immediately
# fetchable from gateways. Production code retries via the urllib3 Retry
# adapter, but for the round-trip test we add an explicit poll so a slow
# propagation doesn't look like a real failure.
_FETCH_POLL_DEADLINE_SECONDS = 180
_FETCH_POLL_INTERVAL_SECONDS = 10


def _make_anchor():
    from ario_mlflow.arweave import ArweaveAnchor

    return ArweaveAnchor(
        os.environ.get("ARIO_MLFLOW_ARWEAVE_WALLET", ""),
        os.environ.get("ARIO_MLFLOW_GATEWAY_HOST", "turbo-gateway.com"),
    )


def _signed_envelope():
    """Mint a tiny real signed envelope — enough bytes to upload meaningfully
    without bloating the test's Turbo footprint."""
    from ario_mlflow.proof import ProofEngine, canonical_json, hash_data

    pe = ProofEngine()
    payload = {"event_type": "live_test", "marker": "ar-io-mlflow:live"}
    payload_bytes = canonical_json(payload)
    env = pe.create_commitment(
        event_type="live_test",
        subject={"type": "smoke_test"},
        payload_bytes=payload_bytes,
        previous_hash="GENESIS",
    )
    return env, payload_bytes, pe


def test_anchor_construction_resolves_wallet():
    """The wallet-resolution path itself must not crash on a real environment.
    A funded wallet is not required for this check — auto-generated is fine."""
    anchor = _make_anchor()
    print(
        f"\n[live] wallet_type={anchor.wallet_type!r} "
        f"wallet_mode={anchor.wallet_mode!r} enabled={anchor.enabled}"
    )
    assert anchor.enabled, (
        f"anchor failed to enable — last_error={anchor.last_error!r}; "
        "check ARIO_MLFLOW_ARWEAVE_WALLET / turbo-sdk install"
    )
    assert anchor.wallet_type in {"solana", "arweave"}


def test_upload_fetch_verify_round_trip():
    """Real end-to-end: sign → Turbo upload → gateway fetch → verify signature.

    Skips cleanly when the wallet is unfunded (``last_error`` mentions
    credits/funding) so an auto-generated keypair doesn't surface as a hard
    failure. The propagation poll gives Arweave's indexing up to
    ``_FETCH_POLL_DEADLINE_SECONDS`` before declaring the fetch broken.
    """
    from ario_mlflow.verify import verify_signature
    from ario_mlflow.proof import ProofEngine

    anchor = _make_anchor()
    envelope, _payload_bytes, _pe = _signed_envelope()

    result = anchor.upload_proof(envelope)
    if result is None:
        err = (anchor.last_error or "").lower()
        if any(s in err for s in ("credit", "balance", "fund", "insufficient")):
            pytest.skip(
                f"wallet is unfunded — Turbo upload returned no credits "
                f"(last_error={anchor.last_error!r}). Fund the wallet to run "
                "this test."
            )
        pytest.fail(
            f"Turbo upload returned None with last_error={anchor.last_error!r}"
        )
    tx_id = result["tx_id"]
    print(f"\n[live] uploaded tx={tx_id} url={result['url']}")

    # Propagation poll — Arweave indexing isn't instantaneous.
    deadline = time.monotonic() + _FETCH_POLL_DEADLINE_SECONDS
    fetched = None
    while time.monotonic() < deadline:
        fetched = anchor.fetch_proof(tx_id)
        if fetched is not None:
            break
        time.sleep(_FETCH_POLL_INTERVAL_SECONDS)
    assert fetched is not None, (
        f"tx {tx_id} not retrievable within {_FETCH_POLL_DEADLINE_SECONDS}s "
        f"across gateways {anchor.gateways!r}; last_error={anchor.last_error!r}"
    )

    # The round-trip envelope must equal what we uploaded, and signature
    # verify must still pass off the gateway-served bytes (catches a gateway
    # that mangles JSON encoding).
    assert fetched["event_id"] == envelope["event_id"]
    assert fetched["payload_hash"] == envelope["payload_hash"]
    assert fetched["signature"] == envelope["signature"]
    assert verify_signature(fetched, ProofEngine())["ok"]


def test_verify_proof_by_tx_against_fresh_upload():
    """The operator-side composite (``verify_proof_by_tx``) must work against
    a freshly uploaded TX — fetches the envelope, runs signature + anchored
    bytes (no MLflow context here → check 2 surfaces ``no_mlflow_client``),
    and ar.io attestation if configured."""
    from ario_mlflow.proof import ProofEngine
    from ario_mlflow.verify import verify_proof_by_tx

    anchor = _make_anchor()
    envelope, _bytes, _pe = _signed_envelope()
    result = anchor.upload_proof(envelope)
    if result is None:
        err = (anchor.last_error or "").lower()
        if any(s in err for s in ("credit", "balance", "fund", "insufficient")):
            pytest.skip(f"wallet unfunded; last_error={anchor.last_error!r}")
        pytest.fail(f"upload returned None; last_error={anchor.last_error!r}")
    tx_id = result["tx_id"]

    # Same propagation poll as the round-trip test.
    deadline = time.monotonic() + _FETCH_POLL_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        if anchor.fetch_proof(tx_id) is not None:
            break
        time.sleep(_FETCH_POLL_INTERVAL_SECONDS)

    out = verify_proof_by_tx(
        tx_id, anchor=anchor, proof_engine=ProofEngine(),
    )
    print(
        f"\n[live] verify_proof_by_tx({tx_id}): "
        f"proof_found={out.get('proof_found')} "
        f"sig={out['signature']['ok']} "
        f"bytes={out['anchored_bytes']['ok']}"
    )
    assert out["proof_found"] is True
    assert out["signature"]["ok"] is True


def test_multi_gateway_fetch_fallback():
    """``fetch_proof`` must walk the gateway list and survive an unreachable
    primary. Doesn't require a fresh upload — uses the well-known marker tx
    from a previous live anchor (the v0.2.0 smoke test). Override with
    ``ARIO_MLFLOW_LIVE_KNOWN_TX`` to use a different TX."""
    from ario_mlflow.arweave import ArweaveAnchor

    known_tx = os.environ.get(
        "ARIO_MLFLOW_LIVE_KNOWN_TX",
        # The v0.2.0 smoke-test envelope. If Arweave ever drops history, this
        # falls back to whatever the env var points at.
        "jwjbN4Td6nNdc-cGovoEGLiCQPThwhWt1TLIoM1JrqA",
    )
    # Bad primary + real fallback. ``ArweaveAnchor.fetch_proof`` doesn't need
    # the wallet to be funded — it's a read-only path — so we point at
    # /dev/null for the wallet to make construction succeed cheaply.
    anchor = ArweaveAnchor(
        "",
        gateway_host="unreachable.example.invalid",
        gateways=["unreachable.example.invalid", "turbo-gateway.com", "ardrive.net"],
    )
    out = anchor.fetch_proof(known_tx)
    print(
        f"\n[live] multi-gateway fetch of {known_tx}: "
        f"got={out is not None} last_error={anchor.last_error!r}"
    )
    if out is None:
        pytest.skip(
            f"known TX {known_tx} not retrievable across fallbacks "
            f"(last_error={anchor.last_error!r}); set ARIO_MLFLOW_LIVE_KNOWN_TX "
            "to a known-good tx to re-enable this check."
        )
    assert "signature" in out or "record" in out, (
        f"gateway returned non-envelope JSON: {sorted(out)}"
    )


def test_ario_attestation_4th_check():
    """The optional 4th verify check — only meaningful when an ar.io Verify
    URL is configured. Polls until level ≥ 1 (or timeout)."""
    verify_url = os.environ.get("ARIO_MLFLOW_ARIO_VERIFY_URL")
    if not verify_url:
        pytest.skip("ARIO_MLFLOW_ARIO_VERIFY_URL not set; ar.io check is opt-in")
    from ario_mlflow.verify import ArioVerifyClient
    from ario_mlflow.proof import ProofEngine

    anchor = _make_anchor()
    envelope, _b, _pe = _signed_envelope()
    upload = anchor.upload_proof(envelope)
    if upload is None:
        err = (anchor.last_error or "").lower()
        if any(s in err for s in ("credit", "balance", "fund", "insufficient")):
            pytest.skip(f"wallet unfunded; last_error={anchor.last_error!r}")
        pytest.fail(f"upload returned None; last_error={anchor.last_error!r}")
    tx_id = upload["tx_id"]

    client = ArioVerifyClient(verify_url)
    # poll_attestation returns the latest attestation result either way.
    result = client.poll_attestation(tx_id, target_level=1, timeout=180, interval=15)
    print(
        f"\n[live] ar.io attestation for {tx_id}: "
        f"level={getattr(result, 'attestation_level', None)} "
        f"last_error={client.last_error!r}"
    )
    assert result is not None, (
        f"ar.io Verify returned no result; last_error={client.last_error!r}"
    )
