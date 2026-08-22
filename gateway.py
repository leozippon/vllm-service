"""Authenticated OpenAI-compatible proxy for one configured vLLM upstream.

The gateway is a shared service for local projects. It intentionally has no
model routing: every ``/v1`` request goes to the same upstream deployment.

- Everything under /v1/ is proxied byte-for-byte, with two deliberate
  exceptions: the Authorization header is replaced with the gateway's own
  upstream credential, and reasoning_effort "high" is mapped to this model's
  top tier "xhigh" (see README).
- Single Bearer key: VLLM_API_KEY from the process environment. systemd loads
  .env through EnvironmentFile=; that is the only place the file is parsed.
- A bounded in-flight cap is a guard rail against runaway clients, not a
  throughput regulator -- the engine scheduler regulates throughput. A
  successful stream holds its slot until the stream ends, the client
  disconnects, or an error occurs.

Run:  uvicorn gateway:app --host 127.0.0.1 --port 8011
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


def _env(name: str) -> str:
    """Read one configuration value from the process environment.

    systemd loads ``.env`` through ``EnvironmentFile=``; that is the single
    place the file is parsed. Manual runs must source it first.
    """

    return os.environ.get(name, "").strip()


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    if not re.fullmatch(r"[+-]?\d+", raw):
        raise SystemExit(f"{name} must be an integer >= {minimum}, got {raw!r}")
    value = int(raw)
    if value < minimum:
        raise SystemExit(f"{name} must be an integer >= {minimum}, got {value}")
    return value


def _env_positive_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(f"{name} must be a number > 0, got {raw!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"{name} must be a number > 0, got {raw!r}")
    return value


DEFAULT_UPSTREAM_BASE = "http://127.0.0.1:8010"
UPSTREAM_BASE = _env("VLLM_GATEWAY_UPSTREAM_URL") or DEFAULT_UPSTREAM_BASE

# Guard rail, not a regulator: the engine scheduler decides throughput, and it
# is prefix-cache and KV aware where the gateway is blind. These bounds only
# stop a runaway client from exhausting the gateway process. Keep
# MAX_CONCURRENCY at or above the engine's --max-num-seqs.
MAX_CONCURRENCY = _env_int("VLLM_GATEWAY_MAX_CONCURRENCY", 32, minimum=1)
MAX_QUEUE = _env_int("VLLM_GATEWAY_MAX_QUEUE", 32, minimum=0)
QUEUE_TIMEOUT_SECONDS = _env_positive_float(
    "VLLM_GATEWAY_QUEUE_TIMEOUT_SECONDS", 300.0
)

# A non-streaming generation emits no bytes until it is complete, so the read
# timeout bounds the whole generation rather than an idle gap. The largest
# plausible single agent turn is ~60,000 output tokens; at ~50 ms per output
# token under the target concurrency that is ~3,000 s, so 3,600 s keeps ~20%
# headroom while staying finite -- a genuinely wedged request still fails.
READ_TIMEOUT_SECONDS = _env_positive_float(
    "VLLM_GATEWAY_READ_TIMEOUT_SECONDS", 3600.0
)
REQUEST_TIMEOUT = httpx.Timeout(
    connect=5.0, read=READ_TIMEOUT_SECONDS, write=120.0, pool=10.0
)
HEALTH_TIMEOUT = httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=5.0)
_RETRY_AFTER = str(max(1, int(QUEUE_TIMEOUT_SECONDS)))

# Header names are lowercase bytes in the ASGI scope.
SKIP_HEADERS = frozenset(
    (
        b"host",
        b"authorization",
        b"content-length",
        b"connection",
        b"transfer-encoding",
        b"accept-encoding",
    )
)

# OpenAI names the model's top reasoning tier "high"; this model's chat
# template names it "xhigh" and rejects "high" outright. Alias the one value
# both enumerations use for the maximum. "minimal" has no counterpart here and
# is deliberately left to fail with the template's own message.
_EFFORT_KEY = b'"reasoning_effort"'
_EFFORT_HIGH = b'"high"'

API_KEY = _env("VLLM_API_KEY")
if not API_KEY:
    raise SystemExit("VLLM_API_KEY is required in the environment")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vllm-gateway")

app = FastAPI(
    title="vLLM Gateway", docs_url=None, redoc_url=None, openapi_url=None
)
client = httpx.AsyncClient(base_url=UPSTREAM_BASE, timeout=REQUEST_TIMEOUT)

_sem = asyncio.Semaphore(MAX_CONCURRENCY)
_admitted = 0
_admit_limit = MAX_CONCURRENCY + MAX_QUEUE


class _QueueFull(Exception):
    pass


class _QueueTimeout(Exception):
    pass


async def _acquire_slot() -> None:
    global _admitted
    if _admitted >= _admit_limit:
        raise _QueueFull
    _admitted += 1
    try:
        await asyncio.wait_for(_sem.acquire(), timeout=QUEUE_TIMEOUT_SECONDS)
    except TimeoutError:
        _admitted -= 1
        raise _QueueTimeout from None
    except BaseException:
        _admitted -= 1
        raise


def _release_slot() -> None:
    global _admitted
    _sem.release()
    _admitted -= 1


def authorized(request: Request) -> bool:
    auth = request.headers.get("authorization", "")
    return auth.lower().startswith("bearer ") and auth[7:].strip() == API_KEY


def _error(
    message: str, err_type: str, status_code: int, **kwargs
) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": err_type,
                "param": None,
                "code": status_code,
            }
        },
        status_code=status_code,
        **kwargs,
    )


def _normalize_effort(body: bytes) -> bytes:
    """Map OpenAI's top reasoning tier onto this model's spelling.

    The scan is a substring test on raw bytes so that ordinary traffic -- which
    already sends "xhigh", and cannot match the leading quote of ``"high"`` --
    never pays for a JSON round trip of a multi-megabyte body.
    """

    if _EFFORT_KEY not in body or _EFFORT_HIGH not in body:
        return body
    try:
        payload = json.loads(body)
    except ValueError:
        return body
    if not isinstance(payload, dict) or payload.get("reasoning_effort") != "high":
        return body
    payload["reasoning_effort"] = "xhigh"
    log.info('reasoning_effort "high" mapped to "xhigh"')
    return json.dumps(payload).encode()


async def _upstream_ready() -> bool:
    try:
        r = await client.get("/health", timeout=HEALTH_TIMEOUT)
    except Exception as exc:
        log.warning("upstream health check failed: %s", exc)
        return False
    if r.status_code == 200:
        return True
    log.warning("upstream health check failed: status %s", r.status_code)
    return False


@app.get("/health/live")
async def health_live() -> Response:
    return JSONResponse({"status": "ok"})


@app.get("/health")
@app.get("/health/ready")
async def health_ready() -> Response:
    if await _upstream_ready():
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "unavailable"}, status_code=503)


class _HeldStreamingResponse(StreamingResponse):
    def __init__(self, content, *, upstream: httpx.Response, **kwargs):
        super().__init__(content, **kwargs)
        self._upstream = upstream

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self._upstream.aclose()
            finally:
                _release_slot()


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH"])
async def proxy_v1(path: str, request: Request) -> Response:
    if not authorized(request):
        return _error(
            "Invalid or missing API key",
            "auth_error",
            401,
            headers={"WWW-Authenticate": "Bearer"},
        )

    body = await request.body()
    if path == "chat/completions":
        body = _normalize_effort(body)

    # Header values are bytes on the wire. Forward them as bytes: httpx encodes
    # str header values as ASCII, so any non-ASCII value would raise.
    headers = [(k, v) for k, v in request.headers.raw if k not in SKIP_HEADERS]
    headers.append((b"authorization", f"Bearer {API_KEY}".encode()))
    # Raw bytes are passed through without content-encoding, so never let the
    # upstream compress.
    headers.append((b"accept-encoding", b"identity"))

    log.info("%s /v1/%s bytes=%d", request.method, path, len(body))

    try:
        upstream_req = client.build_request(
            request.method,
            f"/v1/{path}",
            params=request.query_params,
            headers=headers,
            content=body,
        )
    except httpx.InvalidURL as exc:
        log.warning("invalid request path /v1/%r: %s", path, exc)
        return _error(f"invalid request path: {exc}", "invalid_request_error", 400)

    try:
        await _acquire_slot()
    except _QueueFull:
        log.warning("rejected: gateway queue is full (%d)", _admit_limit)
        return _error(
            "Too many concurrent requests",
            "rate_limit_error",
            429,
            headers={"Retry-After": _RETRY_AFTER},
        )
    except _QueueTimeout:
        log.warning("rejected: timed out waiting for a gateway slot")
        return _error(
            "Timed out waiting for a slot",
            "rate_limit_error",
            429,
            headers={"Retry-After": _RETRY_AFTER},
        )

    hold = True
    upstream_resp = None
    try:
        try:
            streamed = await client.send(upstream_req, stream=True)
        except httpx.TimeoutException as exc:
            # Must precede RequestError: TimeoutException is a subclass of it.
            log.warning("upstream timed out after %.0fs: %s", READ_TIMEOUT_SECONDS, exc)
            return _error(
                f"upstream did not respond within {READ_TIMEOUT_SECONDS:.0f}s",
                "timeout_error",
                504,
            )
        except httpx.RequestError as exc:
            log.warning("upstream unreachable: %s", exc)
            return _error(f"upstream unreachable: {exc}", "upstream_error", 502)

        upstream_resp = streamed
        if streamed.status_code >= 400:
            content = await streamed.aread()
            return Response(
                content=content,
                status_code=streamed.status_code,
                media_type=streamed.headers.get("content-type"),
            )

        passthrough_headers = {
            k: v
            for k, v in streamed.headers.items()
            if k.lower() in ("content-type", "cache-control", "x-request-id")
        }
        is_sse = streamed.headers.get("content-type", "").startswith(
            "text/event-stream"
        )

        async def passthrough():
            try:
                async for chunk in streamed.aiter_raw():
                    yield chunk
            except Exception as exc:
                # Not BaseException: a client disconnect must still cancel us.
                log.warning("upstream stream failed after start: %s", exc)
                if not is_sse:
                    # No safe in-band error frame; let the framing truncate so
                    # the client's transport reports the failure.
                    raise
                error = json.dumps(
                    {
                        "error": {
                            "message": f"upstream stream failed: {exc}",
                            "type": "upstream_error",
                            "param": None,
                            "code": 502,
                        }
                    }
                )
                # Matches what vLLM itself emits on a mid-stream failure.
                yield f"data: {error}\n\ndata: [DONE]\n\n".encode()

        response = _HeldStreamingResponse(
            passthrough(),
            upstream=streamed,
            status_code=streamed.status_code,
            headers=passthrough_headers,
        )
        hold = False
        return response
    finally:
        if hold:
            try:
                if upstream_resp is not None:
                    await upstream_resp.aclose()
            finally:
                _release_slot()
