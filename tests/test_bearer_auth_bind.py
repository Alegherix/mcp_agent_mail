"""Bearer-based agent identity binding tests (OneTUI-hxcr).

These tests cover the new behaviour where:

1. ``BearerAuthMiddleware`` accepts an ``Authorization: Bearer <token>`` whose
   token matches any agent's ``registration_token`` in the database. The
   server-level token continues to authenticate as before.

2. The tool layer treats a request that the middleware identified as agent
   ``X`` as already-bound: ``register_agent(name=X)`` succeeds without an
   explicit ``registration_token`` argument and returns a refreshed
   ``last_active_ts``. ``register_agent(name=Y)`` from the same Bearer
   session still requires Y's own token.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from mcp_agent_mail import config as _config
from mcp_agent_mail.app import build_mcp_server
from mcp_agent_mail.http import BearerAuthMiddleware, build_http_app


# Non-localhost address used so the middleware does NOT short-circuit through
# the localhost bypass when we want to exercise the Bearer path explicitly.
_REMOTE_CLIENT = ("203.0.113.42", 33000)
_ACCEPT_BOTH = "application/json, text/event-stream"


def _rpc(method: str, params: dict[str, Any], rpc_id: str = "1") -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}


def _parse_tool_result(response_text: str) -> dict[str, Any]:
    """Return the ``result`` payload from a tools/call JSON-RPC response.

    Falls back to parsing SSE-style ``data:`` frames when FastMCP picked the
    streaming response shape.
    """
    text = response_text.strip()
    if text.startswith("{"):
        return json.loads(text)
    # SSE: data: {...}\n\n ...
    for line in text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[len("data:") :].strip())
    raise AssertionError(f"unrecognised response body: {response_text!r}")


def _structured(result_payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the structuredContent dict from a tools/call response."""
    result = result_payload.get("result") or {}
    structured = result.get("structuredContent")
    assert isinstance(structured, dict), f"missing structuredContent in {result_payload!r}"
    return structured


async def _register_agent_via_server_token(
    settings,
    app,
    *,
    server_token: str,
    project_key: str,
    agent_name: str,
) -> dict[str, Any]:
    """Bootstrap an agent using the server-level Bearer token from a
    non-localhost address. Returns the structured ``register_agent`` result
    (including ``registration_token``).
    """
    transport = ASGITransport(app=app, client=_REMOTE_CLIENT)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        ensure_resp = await client.post(
            settings.http.path,
            headers={
                "Authorization": f"Bearer {server_token}",
                "Accept": _ACCEPT_BOTH,
            },
            json=_rpc("tools/call", {"name": "ensure_project", "arguments": {"human_key": project_key}}),
        )
        assert ensure_resp.status_code == 200, ensure_resp.text

        reg_resp = await client.post(
            settings.http.path,
            headers={
                "Authorization": f"Bearer {server_token}",
                "Accept": _ACCEPT_BOTH,
            },
            json=_rpc(
                "tools/call",
                {
                    "name": "register_agent",
                    "arguments": {
                        "project_key": project_key,
                        "program": "test-program",
                        "model": "test-model",
                        "name": agent_name,
                    },
                },
            ),
        )
        assert reg_resp.status_code == 200, reg_resp.text
        return _structured(_parse_tool_result(reg_resp.text))


# ---------------------------------------------------------------------------
# Middleware-level tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_accepts_agent_registration_token(isolated_env, monkeypatch):
    """A non-localhost request with ``Bearer <agent_registration_token>``
    must reach the tool layer (not 401), and ``register_agent(name=<same>)``
    must succeed without an explicit ``registration_token`` argument.
    """
    monkeypatch.setenv("HTTP_BEARER_TOKEN", "server-token")
    monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
    monkeypatch.setenv("HTTP_RBAC_ENABLED", "false")
    monkeypatch.setenv("HTTP_RATE_LIMIT_ENABLED", "false")
    with contextlib.suppress(Exception):
        _config.clear_settings_cache()

    settings = _config.get_settings()
    server = build_mcp_server()
    app = build_http_app(settings, server)

    # Step 1: bootstrap an agent (and capture its registration token) using
    # the server-level token. Non-localhost throughout.
    bootstrap = await _register_agent_via_server_token(
        settings,
        app,
        server_token="server-token",
        project_key="/test/bearer/agent",
        agent_name="BlueLake",
    )
    agent_token = bootstrap["registration_token"]
    assert isinstance(agent_token, str) and agent_token
    first_last_active = bootstrap["last_active_ts"]

    # Step 2: present the agent's registration token as the HTTP Bearer
    # credential, with NO `registration_token` in the tool arguments. The
    # request must pass middleware (acceptance #1) and the tool layer must
    # treat the session as bound to BlueLake (acceptance #2).
    transport = ASGITransport(app=app, client=_REMOTE_CLIENT)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            settings.http.path,
            headers={
                "Authorization": f"Bearer {agent_token}",
                "Accept": _ACCEPT_BOTH,
            },
            json=_rpc(
                "tools/call",
                {
                    "name": "register_agent",
                    "arguments": {
                        "project_key": "/test/bearer/agent",
                        "program": "test-program-refreshed",
                        "model": "test-model",
                        "name": "BlueLake",
                    },
                },
            ),
        )
        assert resp.status_code == 200, resp.text
        result = _structured(_parse_tool_result(resp.text))
        assert result["name"] == "BlueLake"
        # The same agent is returned (same id), with last_active_ts refreshed
        # (it must not be older than the bootstrap response).
        assert result["id"] == bootstrap["id"]
        assert result["last_active_ts"] >= first_last_active


@pytest.mark.asyncio
async def test_middleware_rejects_wrong_bearer_from_non_localhost(isolated_env, monkeypatch):
    """A non-localhost request with a Bearer token that matches neither the
    server token nor any agent's registration token must return 401.
    """
    monkeypatch.setenv("HTTP_BEARER_TOKEN", "server-token")
    monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
    monkeypatch.setenv("HTTP_RBAC_ENABLED", "false")
    monkeypatch.setenv("HTTP_RATE_LIMIT_ENABLED", "false")
    with contextlib.suppress(Exception):
        _config.clear_settings_cache()

    settings = _config.get_settings()
    server = build_mcp_server()
    app = build_http_app(settings, server)

    transport = ASGITransport(app=app, client=_REMOTE_CLIENT)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            settings.http.path,
            headers={
                "Authorization": "Bearer not-a-real-token",
                "Accept": _ACCEPT_BOTH,
            },
            json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
        )
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_middleware_still_accepts_server_token(isolated_env, monkeypatch):
    """Server-level Bearer tokens must keep working (no operator regression)."""
    monkeypatch.setenv("HTTP_BEARER_TOKEN", "server-token")
    monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
    monkeypatch.setenv("HTTP_RBAC_ENABLED", "false")
    monkeypatch.setenv("HTTP_RATE_LIMIT_ENABLED", "false")
    with contextlib.suppress(Exception):
        _config.clear_settings_cache()

    settings = _config.get_settings()
    server = build_mcp_server()
    app = build_http_app(settings, server)

    transport = ASGITransport(app=app, client=_REMOTE_CLIENT)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            settings.http.path,
            headers={
                "Authorization": "Bearer server-token",
                "Accept": _ACCEPT_BOTH,
            },
            json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
        )
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_middleware_localhost_bypass_unchanged_without_header(isolated_env, monkeypatch):
    """When no Authorization header is sent and localhost bypass is enabled,
    the request still proceeds. The Bearer extension must not break this.
    """
    monkeypatch.setenv("HTTP_BEARER_TOKEN", "server-token")
    monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "true")
    with contextlib.suppress(Exception):
        _config.clear_settings_cache()

    settings = _config.get_settings()
    server = build_mcp_server()
    app = build_http_app(settings, server)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            settings.http.path,
            headers={"Accept": _ACCEPT_BOTH},
            json=_rpc("tools/call", {"name": "health_check", "arguments": {}}),
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tool-layer tests (per-identity binding)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bearer_session_can_refresh_bound_agent(isolated_env, monkeypatch):
    """End-to-end through the HTTP stack: a Bearer-bound session calling
    ``register_agent(name=<bound>)`` without ``registration_token`` returns
    a refreshed ``last_active_ts`` for the same agent record.
    """
    monkeypatch.setenv("HTTP_BEARER_TOKEN", "server-token")
    monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
    monkeypatch.setenv("HTTP_RBAC_ENABLED", "false")
    monkeypatch.setenv("HTTP_RATE_LIMIT_ENABLED", "false")
    with contextlib.suppress(Exception):
        _config.clear_settings_cache()

    settings = _config.get_settings()
    server = build_mcp_server()
    app = build_http_app(settings, server)

    bootstrap = await _register_agent_via_server_token(
        settings,
        app,
        server_token="server-token",
        project_key="/test/bearer/refresh",
        agent_name="GreenForest",
    )
    agent_token = bootstrap["registration_token"]

    transport = ASGITransport(app=app, client=_REMOTE_CLIENT)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            settings.http.path,
            headers={
                "Authorization": f"Bearer {agent_token}",
                "Accept": _ACCEPT_BOTH,
            },
            json=_rpc(
                "tools/call",
                {
                    "name": "register_agent",
                    "arguments": {
                        "project_key": "/test/bearer/refresh",
                        "program": "test-program",
                        "model": "test-model",
                        "name": "GreenForest",
                    },
                },
            ),
        )
        assert resp.status_code == 200, resp.text
        result = _structured(_parse_tool_result(resp.text))
        assert result["id"] == bootstrap["id"]
        assert result["name"] == "GreenForest"
        # last_active_ts is refreshed (>= the bootstrap value).
        assert result["last_active_ts"] >= bootstrap["last_active_ts"]


@pytest.mark.asyncio
async def test_bearer_session_a_cannot_skip_token_for_agent_b(isolated_env, monkeypatch):
    """A Bearer session authenticated as agent A must NOT be able to call
    ``register_agent(name=B)`` without B's own registration token. Binding
    is per-identity, not blanket admin.
    """
    monkeypatch.setenv("HTTP_BEARER_TOKEN", "server-token")
    monkeypatch.setenv("HTTP_ALLOW_LOCALHOST_UNAUTHENTICATED", "false")
    monkeypatch.setenv("HTTP_RBAC_ENABLED", "false")
    monkeypatch.setenv("HTTP_RATE_LIMIT_ENABLED", "false")
    with contextlib.suppress(Exception):
        _config.clear_settings_cache()

    settings = _config.get_settings()
    server = build_mcp_server()
    app = build_http_app(settings, server)

    # Register two agents in the same project.
    agent_a = await _register_agent_via_server_token(
        settings,
        app,
        server_token="server-token",
        project_key="/test/bearer/cross",
        agent_name="RedRiver",
    )
    agent_b = await _register_agent_via_server_token(
        settings,
        app,
        server_token="server-token",
        project_key="/test/bearer/cross",
        agent_name="SilverPeak",
    )
    a_token = agent_a["registration_token"]
    assert isinstance(a_token, str) and a_token
    assert agent_b["id"] != agent_a["id"]

    # As Bearer-A, attempt to register/refresh B without a token. The HTTP
    # request succeeds (middleware lets it through; identity is A), but the
    # tool layer must raise AUTHENTICATION_REQUIRED for B.
    transport = ASGITransport(app=app, client=_REMOTE_CLIENT)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            settings.http.path,
            headers={
                "Authorization": f"Bearer {a_token}",
                "Accept": _ACCEPT_BOTH,
            },
            json=_rpc(
                "tools/call",
                {
                    "name": "register_agent",
                    "arguments": {
                        "project_key": "/test/bearer/cross",
                        "program": "test-program",
                        "model": "test-model",
                        "name": "SilverPeak",
                    },
                },
            ),
        )
        # The middleware accepted the request (status 200 for the JSON-RPC
        # envelope). The tool layer raises a structured ToolExecutionError
        # whose payload mentions registration_token / AUTHENTICATION_REQUIRED.
        assert resp.status_code == 200, resp.text
        payload = _parse_tool_result(resp.text)
        result = payload.get("result") or {}
        # Either isError=True with a structured error, or a JSON-RPC error
        # field at the envelope level. Either way the response must not be
        # a success that returned B's profile.
        if "error" in payload:
            err = payload["error"]
            assert "registration_token" in json.dumps(err).lower() or "authentication" in json.dumps(err).lower()
        else:
            assert result.get("isError") is True, payload
            text_blob = json.dumps(result)
            assert "registration_token" in text_blob.lower() or "authentication" in text_blob.lower()


# ---------------------------------------------------------------------------
# Helper unit test: token extraction
# ---------------------------------------------------------------------------


def test_extract_bearer_token_parses_header_or_returns_none():
    assert BearerAuthMiddleware._extract_bearer_token("Bearer abcdef") == "abcdef"
    assert BearerAuthMiddleware._extract_bearer_token("bearer abcdef") == "abcdef"
    assert BearerAuthMiddleware._extract_bearer_token("Bearer   abcdef  ") == "abcdef"
    assert BearerAuthMiddleware._extract_bearer_token("") is None
    assert BearerAuthMiddleware._extract_bearer_token("abcdef") is None
    assert BearerAuthMiddleware._extract_bearer_token("Basic abcdef") is None
    assert BearerAuthMiddleware._extract_bearer_token("Bearer ") is None
