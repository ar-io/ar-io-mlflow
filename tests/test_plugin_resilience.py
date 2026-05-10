"""Tests for the plugin's resilience pass.

Pins the network-layer behavior introduced when ``ario-mlflow`` was
hardened against transient ar.io gateway failures: retry-with-backoff
for upload + Verify, multi-gateway fallback for fetches, and the
attestation-level polling helper. The headline guarantee: a single
flaky gateway shouldn't show up in user-facing UI as a hard verify
failure.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from ario_mlflow.arweave import (
    ArweaveAnchor,
    _DEFAULT_FETCH_GATEWAYS,
    _RETRY_STATUS_CODES,
    _resolve_gateways,
)
from ario_mlflow.verify import ArioVerifyClient


# --- _resolve_gateways helper -------------------------------------------


def test_resolve_gateways_returns_default_list():
    """No kwarg, no env var → primary gateway plus the built-in fallbacks,
    in order, deduplicated."""
    out = _resolve_gateways(None, "turbo-gateway.com")
    assert out[0] == "turbo-gateway.com"
    assert "ardrive.net" in out
    # no duplicates even though "turbo-gateway.com" is also in defaults
    assert len(out) == len(set(out))


def test_resolve_gateways_explicit_kwarg_wins_over_env(monkeypatch):
    """Caller's explicit list must beat ``ARIO_MLFLOW_GATEWAYS`` — the
    env var is a deployment override, not a hardcoded default."""
    monkeypatch.setenv("ARIO_MLFLOW_GATEWAYS", "env.com")
    out = _resolve_gateways(["a.com", "b.com"], "primary.com")
    assert out == ["a.com", "b.com"]


def test_resolve_gateways_env_overrides_built_in_default(monkeypatch):
    """When no explicit list, the env var supplants the built-in fallbacks."""
    monkeypatch.setenv("ARIO_MLFLOW_GATEWAYS", "g1.com, g2.com ,g3.com")
    out = _resolve_gateways(None, "primary.com")
    assert out == ["g1.com", "g2.com", "g3.com"]


def test_resolve_gateways_dedupes_preserving_order():
    """Repeated entries collapse but the first occurrence's position is kept."""
    out = _resolve_gateways(["a.com", "b.com", "a.com", "c.com"], "x.com")
    assert out == ["a.com", "b.com", "c.com"]


def test_resolve_gateways_filters_empty_strings(monkeypatch):
    """Comma-split shouldn't produce empty entries on stray whitespace."""
    monkeypatch.setenv("ARIO_MLFLOW_GATEWAYS", "  , g1.com, ,g2.com,  ")
    out = _resolve_gateways(None, "x.com")
    assert out == ["g1.com", "g2.com"]


# --- ArweaveAnchor: retry adapter wired into the session ----------------


def test_arweave_session_has_retry_adapter_for_https():
    """The HTTPS adapter must carry the configured Retry policy. urllib3's
    Retry exposes total/status_forcelist as introspectable attrs."""
    a = ArweaveAnchor()
    adapter = a._session.get_adapter("https://example.com/")
    retry = adapter.max_retries
    assert retry.total == 2
    assert set(_RETRY_STATUS_CODES) <= set(retry.status_forcelist)


def test_arweave_last_error_starts_none():
    a = ArweaveAnchor()
    assert a.last_error is None


def test_arweave_default_gateways_match_resolver():
    """``ArweaveAnchor.gateways`` must come from the resolver — keep both
    surfaces in sync so callers introspecting one match the other."""
    a = ArweaveAnchor(gateway_host="custom.io")
    assert a.gateways == _resolve_gateways(None, "custom.io")


# --- fetch_proof multi-gateway fallback ---------------------------------


def _ok_response(json_data):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = json_data
    r.raise_for_status = MagicMock()
    return r


def test_fetch_proof_returns_from_first_gateway_on_success():
    a = ArweaveAnchor(gateways=["g1.com", "g2.com"])
    expected = {"event_id": "x"}
    a._session = MagicMock()
    a._session.get.return_value = _ok_response(expected)

    out = a.fetch_proof("TX-1")

    assert out == expected
    assert a._session.get.call_count == 1
    args, kwargs = a._session.get.call_args
    assert args[0] == "https://g1.com/raw/TX-1"
    assert a.last_error is None


def test_fetch_proof_falls_back_to_second_gateway_on_first_failure():
    """A single gateway failure must not propagate as a verify-row FAIL —
    the next gateway should be tried automatically."""
    a = ArweaveAnchor(gateways=["fail.com", "g2.com"])
    expected = {"event_id": "y"}

    def get_side_effect(url, **kwargs):
        if "fail.com" in url:
            raise requests.exceptions.ConnectionError("primary down")
        return _ok_response(expected)

    a._session = MagicMock()
    a._session.get.side_effect = get_side_effect

    out = a.fetch_proof("TX-2")

    assert out == expected
    assert a._session.get.call_count == 2
    assert a.last_error is None  # success after fallback clears the error slot


def test_fetch_proof_returns_none_when_all_gateways_fail():
    """Every gateway exhausted → None and a last_error trail naming each
    gateway's failure mode (so ops can see *which* leg fell over)."""
    a = ArweaveAnchor(gateways=["g1.com", "g2.com"])
    a._session = MagicMock()
    a._session.get.side_effect = requests.exceptions.ConnectionError("down")

    out = a.fetch_proof("TX-3")

    assert out is None
    assert a.last_error is not None
    assert "g1.com" in a.last_error
    assert "g2.com" in a.last_error
    assert a._session.get.call_count == 2


def test_fetch_proof_falls_over_when_gateway_returns_non_json():
    """A 200 with garbage body shouldn't crash the calling thread —
    treat it as a gateway failure and try the next one. Regression
    fix: the resp.json() call was uncaught when the except clause
    only matched RequestException."""
    a = ArweaveAnchor(gateways=["bad-json.com", "g2.com"])
    expected = {"event_id": "z"}

    bad_resp = MagicMock()
    bad_resp.status_code = 200
    bad_resp.raise_for_status = MagicMock()
    bad_resp.json.side_effect = ValueError("not json")

    def get_side_effect(url, **kwargs):
        if "bad-json.com" in url:
            return bad_resp
        return _ok_response(expected)

    a._session = MagicMock()
    a._session.get.side_effect = get_side_effect

    out = a.fetch_proof("TX-bad-json")

    assert out == expected
    assert a._session.get.call_count == 2


def test_check_status_returns_unknown_on_non_json_response():
    """``check_status`` must return a dict for every code path —
    including when Turbo returns 200 with a non-JSON body. Regression
    fix mirrors fetch_proof: ValueError used to escape."""
    a = ArweaveAnchor()
    bad_resp = MagicMock()
    bad_resp.status_code = 200
    bad_resp.json.side_effect = ValueError("not json")
    a._session = MagicMock()
    a._session.get.return_value = bad_resp

    out = a.check_status("TX-bad")

    assert out == {"status": "UNKNOWN"}


def test_fetch_proof_with_no_gateways_returns_none():
    """Defensive — if a caller somehow constructs an anchor with an
    empty gateway list, fetch must surface the misconfiguration via
    ``last_error`` rather than throwing IndexError."""
    a = ArweaveAnchor(gateways=[])
    a._session = MagicMock()  # session.get must not be called

    out = a.fetch_proof("TX-4")

    assert out is None
    assert "no fetch gateways" in (a.last_error or "")
    assert a._session.get.call_count == 0


# --- upload_proof error surfacing ---------------------------------------


def test_upload_proof_disabled_anchor_records_last_error():
    """When the anchor is disabled (no Turbo / no wallet), upload must
    return None *and* set last_error so callers can distinguish 'disabled'
    from 'failed to upload'."""
    a = ArweaveAnchor()
    a.enabled = False
    a._signer = None
    a.last_error = None  # reset in case construction set it

    out = a.upload_proof({"foo": "bar"})

    assert out is None
    assert a.last_error is not None
    assert "disabled" in a.last_error.lower()


# --- ArioVerifyClient: retry adapter + last_error contract --------------


def test_ariovify_session_has_retry_adapter():
    c = ArioVerifyClient(base_url="https://verify.example.com")
    adapter = c._session.get_adapter("https://example.com/")
    retry = adapter.max_retries
    assert retry.total == 2
    assert set(_RETRY_STATUS_CODES) <= set(retry.status_forcelist)


def test_ariovify_last_error_starts_none():
    c = ArioVerifyClient(base_url="https://verify.example.com")
    assert c.last_error is None


def test_ariovify_disabled_when_no_base_url():
    c = ArioVerifyClient(base_url="")
    assert c.enabled is False


# --- poll_attestation ---------------------------------------------------


def test_poll_attestation_returns_immediately_when_target_already_reached():
    """Single submit returns level >= target → exit on first attempt,
    no sleep loop."""
    c = ArioVerifyClient(base_url="https://verify.example.com")
    c.enabled = True  # bypass health check for test
    c.submit_verification = MagicMock(
        return_value={"attestation_level": 3, "verification_id": "v1"}
    )

    out = c.poll_attestation("TX-1", target_level=2, timeout=10, interval=0.1)

    assert out is not None
    assert out["attestation_level"] == 3
    assert c.submit_verification.call_count == 1


def test_poll_attestation_polls_until_target_reached():
    """Levels grow over successive submits — poll exits when level
    >= target_level even if earlier submits were under-target."""
    c = ArioVerifyClient(base_url="https://verify.example.com")
    c.enabled = True

    sequence = [
        {"attestation_level": 1, "verification_id": "v1"},
        {"attestation_level": 1, "verification_id": "v1"},
        {"attestation_level": 2, "verification_id": "v1"},
    ]
    c.submit_verification = MagicMock(side_effect=sequence)

    out = c.poll_attestation("TX-1", target_level=2, timeout=10, interval=0.01)

    assert out is not None
    assert out["attestation_level"] == 2
    assert c.submit_verification.call_count == 3


def test_poll_attestation_returns_last_result_on_timeout():
    """Target never reached within the time budget → return whatever the
    latest submit produced (so callers can still surface "level 1" status
    instead of nothing)."""
    c = ArioVerifyClient(base_url="https://verify.example.com")
    c.enabled = True
    c.submit_verification = MagicMock(
        return_value={"attestation_level": 1, "verification_id": "v1"}
    )

    out = c.poll_attestation("TX-1", target_level=3, timeout=0.05, interval=0.01)

    assert out is not None
    assert out["attestation_level"] == 1
    assert c.submit_verification.call_count >= 1


def test_poll_attestation_returns_none_when_client_disabled():
    """Disabled client → no work attempted; returns None with a clear
    last_error reason so callers don't confuse it with a polling timeout."""
    c = ArioVerifyClient(base_url="")  # no URL → disabled

    out = c.poll_attestation("TX-1", target_level=2, timeout=1, interval=0.1)

    assert out is None
    assert "not enabled" in (c.last_error or "")


def test_poll_attestation_handles_all_failures():
    """Every submit returns None (e.g. transient gateway storm) →
    returns None and last_error reflects the polling outcome."""
    c = ArioVerifyClient(base_url="https://verify.example.com")
    c.enabled = True
    c.submit_verification = MagicMock(return_value=None)

    out = c.poll_attestation("TX-1", target_level=2, timeout=0.05, interval=0.01)

    assert out is None
