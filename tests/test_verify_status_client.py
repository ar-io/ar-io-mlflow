"""Tests for VerifyStatusClient against an in-process HTTP stub.

No network, no live agent — a loopback ``ThreadingHTTPServer`` plays the
agent (or api-guard proxy) per `verify-status-api.md`. Follows the repo
convention of zero extra test dependencies.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ario_mlflow.errors import (
    AssetVerificationError,
    VerifyStatusAuthError,
    VerifyStatusError,
    VerifyStatusLicenseError,
    VerifyStatusTransportError,
    VerifyStatusUnknownAssetError,
)
from ario_mlflow.verify_status_client import VerifyStatus, VerifyStatusClient

SECRET = "test-management-secret"

GOOD_BODY = {
    "asset_id": "customer-pii-models",
    "tenant_id": "acme-prod",
    "agent_id": "data-warehouse-01",
    "outcome": "verified",
    "last_verified_at": "2026-06-10T14:23:11.420Z",
    "max_age": 3600,
    "stale": False,
    "policy_hash": "a3f1b2c4d5e6",
    "current_tx_id": "kJ7vX2pQ8mNcLw9Yb4Tz_Fh1aGdRsE6oWp",
}


class _StubAgent:
    """Configurable loopback stand-in for the agent's management port."""

    def __init__(self):
        self.status_code = 200
        self.body: object = GOOD_BODY
        self.raw_body: bytes | None = None  # overrides body when set
        self.requests: list[dict] = []  # {"path": ..., "headers": ...} per hit

        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                stub.requests.append(
                    {"path": self.path, "headers": dict(self.headers)}
                )
                payload = (
                    stub.raw_body
                    if stub.raw_body is not None
                    else json.dumps(stub.body).encode()
                )
                self.send_response(stub.status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):  # keep pytest output clean
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub_agent():
    stub = _StubAgent()
    yield stub
    stub.close()


@pytest.fixture
def client(stub_agent):
    return VerifyStatusClient(stub_agent.base_url, secret=SECRET)


# --- constructor ------------------------------------------------------------


def test_requires_exactly_one_credential():
    with pytest.raises(ValueError):
        VerifyStatusClient("http://127.0.0.1:9847")
    with pytest.raises(ValueError):
        VerifyStatusClient("http://127.0.0.1:9847", secret="s", api_key="k")


# --- 200 happy path ---------------------------------------------------------


def test_get_parses_contract_fields(client):
    status = client.get("customer-pii-models")
    assert status == VerifyStatus(
        asset_id="customer-pii-models",
        tenant_id="acme-prod",
        agent_id="data-warehouse-01",
        outcome="verified",
        last_verified_at="2026-06-10T14:23:11.420Z",
        max_age=3600,
        stale=False,
        policy_hash="a3f1b2c4d5e6",
        current_tx_id="kJ7vX2pQ8mNcLw9Yb4Tz_Fh1aGdRsE6oWp",
    )


def test_management_secret_header_sent(stub_agent, client):
    client.get("customer-pii-models")
    headers = stub_agent.requests[0]["headers"]
    assert headers.get("X-Ario-Management-Secret") == SECRET
    assert "Authorization" not in headers


def test_api_key_form_sends_bearer(stub_agent):
    proxied = VerifyStatusClient(stub_agent.base_url, api_key="ario_test_key")
    proxied.get("customer-pii-models")
    headers = stub_agent.requests[0]["headers"]
    assert headers.get("Authorization") == "Bearer ario_test_key"
    assert "X-Ario-Management-Secret" not in headers


def test_asset_id_is_percent_encoded(stub_agent, client):
    client.get("models/fraud v2")
    assert stub_agent.requests[0]["path"] == "/v1/verify-status/models%2Ffraud%20v2"


def test_unknown_response_fields_ignored(stub_agent, client):
    stub_agent.body = {**GOOD_BODY, "future_field": "ignore me"}
    assert client.get("customer-pii-models").outcome == "verified"


def test_unrecognized_outcome_normalized_to_unknown(stub_agent, client):
    stub_agent.body = {**GOOD_BODY, "outcome": "quarantined"}
    assert client.get("customer-pii-models").outcome == "unknown"


def test_null_fields_parse(stub_agent, client):
    stub_agent.body = {
        **GOOD_BODY,
        "outcome": "unknown",
        "last_verified_at": None,
        "stale": True,
        "current_tx_id": None,
    }
    status = client.get("customer-pii-models")
    assert status.last_verified_at is None
    assert status.current_tx_id is None
    assert status.stale is True


# --- error model (contract §7): status codes only, never message strings ----


def test_401_raises_auth_error(stub_agent, client):
    stub_agent.status_code = 401
    stub_agent.body = {"error": "unauthorized: missing or invalid X-Ario-Management-Secret"}
    with pytest.raises(VerifyStatusAuthError) as exc_info:
        client.get("customer-pii-models")
    assert exc_info.value.asset_id == "customer-pii-models"


def test_404_raises_unknown_asset_error(stub_agent, client):
    stub_agent.status_code = 404
    stub_agent.body = {"error": "asset_id not in policy"}
    with pytest.raises(VerifyStatusUnknownAssetError):
        client.get("not-in-policy")


def test_503_raises_license_error_with_upgrade_url(stub_agent, client):
    stub_agent.status_code = 503
    stub_agent.body = {"error": "license required", "upgrade_url": "https://ar.io/upgrade"}
    with pytest.raises(VerifyStatusLicenseError) as exc_info:
        client.get("customer-pii-models")
    assert exc_info.value.upgrade_url == "https://ar.io/upgrade"
    assert "https://ar.io/upgrade" in str(exc_info.value)
    # License gating is a transport-level refusal, not a verification verdict.
    assert isinstance(exc_info.value, VerifyStatusTransportError)


def test_503_without_body_url_still_license_error(stub_agent, client):
    stub_agent.status_code = 503
    stub_agent.raw_body = b"Service Unavailable"
    with pytest.raises(VerifyStatusLicenseError) as exc_info:
        client.get("customer-pii-models")
    assert exc_info.value.upgrade_url is None


def test_500_raises_transport_error(stub_agent, client):
    stub_agent.status_code = 500
    stub_agent.body = {"error": "state.db read failed"}
    with pytest.raises(VerifyStatusTransportError):
        client.get("customer-pii-models")


def test_connection_refused_raises_transport_error(stub_agent):
    stub_agent.close()  # nothing listening anymore
    client = VerifyStatusClient(stub_agent.base_url, secret=SECRET, timeout=0.5)
    with pytest.raises(VerifyStatusTransportError):
        client.get("customer-pii-models")


def test_malformed_200_body_raises_transport_error(stub_agent, client):
    stub_agent.raw_body = b"not json"
    with pytest.raises(VerifyStatusTransportError):
        client.get("customer-pii-models")


def test_missing_required_field_raises_transport_error(stub_agent, client):
    body = dict(GOOD_BODY)
    del body["policy_hash"]
    stub_agent.body = body
    with pytest.raises(VerifyStatusTransportError):
        client.get("customer-pii-models")


def test_all_errors_share_the_family_bases(stub_agent, client):
    stub_agent.status_code = 401
    with pytest.raises(VerifyStatusError):
        client.get("a")
    with pytest.raises(AssetVerificationError):
        client.get("a")


# --- caching (contract §6.1 / §9.2) ------------------------------------------


def test_default_get_always_refetches(stub_agent, client):
    client.get("customer-pii-models")
    client.get("customer-pii-models")
    assert len(stub_agent.requests) == 2


def test_max_cache_age_serves_from_cache(stub_agent, client):
    first = client.get("customer-pii-models")
    second = client.get("customer-pii-models", max_cache_age=60.0)
    assert len(stub_agent.requests) == 1
    assert second == first


def test_cache_is_per_asset(stub_agent, client):
    stub_agent.body = {**GOOD_BODY, "asset_id": "asset-a"}
    client.get("asset-a")
    stub_agent.body = {**GOOD_BODY, "asset_id": "asset-b"}
    client.get("asset-b", max_cache_age=60.0)  # not cached yet → fetches
    assert len(stub_agent.requests) == 2


def test_expired_cache_refetches(stub_agent, client, monkeypatch):
    import ario_mlflow.verify_status_client as vsc

    t = {"now": 1000.0}
    monkeypatch.setattr(vsc.time, "monotonic", lambda: t["now"])
    client.get("customer-pii-models")
    t["now"] += 31.0
    client.get("customer-pii-models", max_cache_age=30.0)
    assert len(stub_agent.requests) == 2
