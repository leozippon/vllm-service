"""Gateway authentication, header handling, admission control, and streaming."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGURED_UPSTREAM = "http://configured-upstream.invalid:9000"

# Baked process-wide on purpose: the admission-control tests need small bounds,
# and gateway configuration is read once at import. The production defaults are
# covered in a subprocess instead (see ConfigurationTests).
os.environ["VLLM_API_KEY"] = "test-key"
os.environ["VLLM_GATEWAY_UPSTREAM_URL"] = CONFIGURED_UPSTREAM
os.environ["VLLM_GATEWAY_MAX_CONCURRENCY"] = "1"
os.environ["VLLM_GATEWAY_MAX_QUEUE"] = "1"
os.environ["VLLM_GATEWAY_QUEUE_TIMEOUT_SECONDS"] = "0.3"

import httpx

import gateway

AUTH = {"Authorization": "Bearer test-key"}


class _Blocked(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected upstream call: {request.url}")


gateway.client = httpx.AsyncClient(
    transport=_Blocked(),
    base_url="http://upstream.invalid",
)


class _Upstream(httpx.AsyncBaseTransport):
    def __init__(self, *, health_status: int = 200, fail_connect: bool = False) -> None:
        self.health_status = health_status
        self.fail_connect = fail_connect
        self.block = asyncio.Event()
        self.entered = asyncio.Event()
        self.in_flight = 0
        self.max_seen = 0
        self.closed = False
        self.health_timeout = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.fail_connect:
            raise httpx.ConnectError("upstream down", request=request)
        if request.url.path == "/health":
            self.health_timeout = request.extensions.get("timeout")
            return httpx.Response(self.health_status, request=request)

        transport = self

        class _Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                transport.in_flight += 1
                transport.max_seen = max(transport.max_seen, transport.in_flight)
                transport.entered.set()
                try:
                    yield b"A"
                    await transport.block.wait()
                    yield b"B"
                finally:
                    transport.in_flight -= 1

            async def aclose(self):
                transport.closed = True

        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            stream=_Body(),
            request=request,
        )


class _Bytes(httpx.AsyncByteStream):
    """A response body the gateway can stream, as a real transport would return."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def __aiter__(self):
        yield self._data

    async def aclose(self) -> None:
        return None


class _Recorder(httpx.AsyncBaseTransport):
    """Records what actually reached the upstream and replies with a fixed body."""

    def __init__(
        self,
        *,
        status: int = 200,
        content: bytes = b'{"ok": true}',
        content_type: str = "application/json",
    ) -> None:
        self.status = status
        self.content = content
        self.content_type = content_type
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        return httpx.Response(
            self.status,
            headers={"content-type": self.content_type},
            stream=_Bytes(self.content),
            request=request,
        )

    @property
    def last(self) -> httpx.Request:
        if len(self.requests) != 1:
            raise AssertionError(f"expected exactly one upstream call: {self.requests}")
        return self.requests[0]


class _Raiser(httpx.AsyncBaseTransport):
    def __init__(self, exc_factory) -> None:
        self.exc_factory = exc_factory

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise self.exc_factory(request)


class _MidStreamFailure(httpx.AsyncBaseTransport):
    def __init__(self, content_type: str) -> None:
        self.content_type = content_type
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = self

        class _Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"A"
                raise RuntimeError("upstream died")

            async def aclose(self):
                transport.closed = True

        return httpx.Response(
            200,
            headers={"content-type": self.content_type},
            stream=_Body(),
            request=request,
        )


def _install(upstream: httpx.AsyncBaseTransport) -> None:
    gateway.client = httpx.AsyncClient(
        transport=upstream,
        base_url="http://upstream.invalid",
        timeout=httpx.Timeout(5.0),
    )


def _scope(path: str, headers: list[tuple[bytes, bytes]], method: str = "POST") -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("latin-1"),
        "query_string": b"",
        "headers": [(b"host", b"gateway.invalid")] + headers,
        "client": ("127.0.0.1", 123),
        "server": ("gateway.invalid", 80),
        "root_path": "",
    }


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        gateway._admitted = 0
        gateway._sem = asyncio.Semaphore(gateway.MAX_CONCURRENCY)

    async def asyncTearDown(self) -> None:
        await gateway.client.aclose()
        gateway.client = httpx.AsyncClient(
            transport=_Blocked(),
            base_url="http://upstream.invalid",
        )

    def _gw(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway.app),
            base_url="http://gateway.invalid",
        )

    async def test_stream_holds_slot_until_done(self) -> None:
        up = _Upstream()
        _install(up)
        async with self._gw() as client:
            async def once() -> None:
                response = await client.post(
                    "/v1/chat/completions", headers=AUTH, content=b"{}"
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, b"AB")

            first = asyncio.create_task(once())
            await asyncio.wait_for(up.entered.wait(), timeout=1)
            second = asyncio.create_task(once())
            await asyncio.sleep(0.05)
            self.assertEqual(up.max_seen, 1)
            self.assertEqual(up.in_flight, 1)
            self.assertFalse(second.done())
            up.block.set()
            await asyncio.gather(first, second)
            self.assertEqual(up.max_seen, 1)
            self.assertEqual(gateway._admitted, 0)

        await gateway.client.aclose()
        up = _Upstream()
        _install(up)
        request_sent = False

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": b"{}", "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                await asyncio.Event().wait()

        scope = _scope(
            "/v1/chat/completions", [(b"authorization", b"Bearer test-key")]
        )
        await asyncio.wait_for(gateway.app(scope, receive, send), timeout=1)
        self.assertEqual(gateway._admitted, 0)
        self.assertTrue(up.closed)

    async def test_queue_full_and_timeout_return_429(self) -> None:
        up = _Upstream()
        _install(up)
        async with self._gw() as client:
            async def hold() -> None:
                async with client.stream(
                    "POST", "/v1/chat/completions", headers=AUTH, content=b"{}"
                ) as response:
                    async for _ in response.aiter_bytes():
                        pass

            holder = asyncio.create_task(hold())
            try:
                await asyncio.wait_for(up.entered.wait(), timeout=1)
                queued = asyncio.create_task(
                    client.post(
                        "/v1/chat/completions", headers=AUTH, content=b"{}"
                    )
                )
                for _ in range(100):
                    if gateway._admitted >= 2:
                        break
                    await asyncio.sleep(0.01)
                else:
                    self.fail(f"queued request was not admitted: {gateway._admitted}")

                full = await client.post(
                    "/v1/chat/completions", headers=AUTH, content=b"{}"
                )
                self.assertEqual(full.status_code, 429)
                self.assertIn("retry-after", full.headers)
                self.assertEqual(full.json()["error"]["type"], "rate_limit_error")

                timed_out = await asyncio.wait_for(queued, timeout=1)
                self.assertEqual(timed_out.status_code, 429)
                self.assertIn("retry-after", timed_out.headers)
            finally:
                up.block.set()
                await asyncio.wait_for(holder, timeout=1)

    async def test_liveness_and_readiness(self) -> None:
        self.assertEqual(gateway.DEFAULT_UPSTREAM_BASE, "http://127.0.0.1:8010")
        self.assertEqual(gateway.UPSTREAM_BASE, CONFIGURED_UPSTREAM)
        async with self._gw() as client:
            healthy = _Upstream(health_status=200)
            _install(healthy)
            self.assertEqual((await client.get("/health/live")).status_code, 200)
            self.assertEqual((await client.get("/health/ready")).status_code, 200)
            self.assertEqual((await client.get("/health")).status_code, 200)
            # The probe must not inherit the multi-hour generation read timeout.
            self.assertEqual(healthy.health_timeout, gateway.HEALTH_TIMEOUT.as_dict())
            self.assertNotEqual(
                gateway.HEALTH_TIMEOUT.as_dict(), gateway.REQUEST_TIMEOUT.as_dict()
            )

            _install(_Upstream(health_status=500))
            self.assertEqual((await client.get("/health/ready")).status_code, 503)
            self.assertEqual((await client.get("/health")).status_code, 503)
            self.assertEqual((await client.get("/health/live")).status_code, 200)

            _install(_Upstream(fail_connect=True))
            down = await client.get("/health/ready")
            self.assertEqual(down.status_code, 503)
            self.assertEqual(down.json(), {"status": "unavailable"})
            self.assertEqual((await client.get("/health")).status_code, 503)
            self.assertEqual((await client.get("/health/live")).status_code, 200)

    async def test_authentication_rejects_bad_credentials(self) -> None:
        # gateway.client is the _Blocked transport here: it asserts that no
        # rejected request ever reaches the upstream.
        async with self._gw() as client:
            for headers in (
                {},
                {"Authorization": "Bearer wrong"},
                {"Authorization": "Basic test-key"},
                {"Authorization": "Bearer "},
                {"Authorization": "test-key"},
            ):
                with self.subTest(headers=headers):
                    r = await client.post(
                        "/v1/chat/completions", headers=headers, content=b"{}"
                    )
                    self.assertEqual(r.status_code, 401)
                    self.assertEqual(r.headers["www-authenticate"], "Bearer")
                    self.assertEqual(
                        r.json()["error"],
                        {
                            "message": "Invalid or missing API key",
                            "type": "auth_error",
                            "param": None,
                            "code": 401,
                        },
                    )
                    self.assertEqual(gateway._admitted, 0)

        recorder = _Recorder()
        _install(recorder)
        async with self._gw() as client:
            accepted = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "bearer test-key"},
                content=b"{}",
            )
            self.assertEqual(accepted.status_code, 200)
        self.assertEqual(len(recorder.requests), 1)

    async def test_client_authorization_is_replaced(self) -> None:
        recorder = _Recorder()
        _install(recorder)
        async with self._gw() as client:
            # The first credential authenticates the client; a second one, and
            # the client's own accept-encoding, must never reach the upstream.
            r = await client.post(
                "/v1/chat/completions",
                headers=[
                    (b"authorization", b"bearer test-key"),
                    (b"authorization", b"Bearer leaked-secret"),
                    (b"x-custom", b"keep"),
                    (b"accept-encoding", b"gzip"),
                ],
                content=b"{}",
            )
            self.assertEqual(r.status_code, 200)

        raw = [(k.lower(), v) for k, v in recorder.last.headers.raw]
        auth = [v for k, v in raw if k == b"authorization"]
        self.assertEqual(auth, [f"Bearer {gateway.API_KEY}".encode()])
        self.assertNotIn(b"leaked-secret", b"|".join(v for _, v in raw))
        self.assertEqual([v for k, v in raw if k == b"accept-encoding"], [b"identity"])
        self.assertIn((b"x-custom", b"keep"), raw)
        self.assertEqual(gateway._admitted, 0)

    async def test_non_ascii_header_is_forwarded(self) -> None:
        recorder = _Recorder()
        _install(recorder)
        async with self._gw() as client:
            r = await client.post(
                "/v1/chat/completions",
                headers=[
                    (b"authorization", b"Bearer test-key"),
                    (b"x-title", "café".encode()),
                ],
                content=b"{}",
            )
            self.assertEqual(r.status_code, 200)
        self.assertIn(
            (b"x-title", "café".encode()),
            [(k.lower(), v) for k, v in recorder.last.headers.raw],
        )
        self.assertEqual(gateway._admitted, 0)

    async def test_invalid_path_returns_400(self) -> None:
        async with self._gw() as client:
            r = await client.get("/v1/%00", headers=AUTH)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["error"]["type"], "invalid_request_error")
        self.assertEqual(r.json()["error"]["code"], 400)
        self.assertEqual(gateway._admitted, 0)

    async def test_upstream_timeout_returns_504(self) -> None:
        _install(_Raiser(lambda req: httpx.ReadTimeout("slow", request=req)))
        async with self._gw() as client:
            r = await client.post(
                "/v1/chat/completions", headers=AUTH, content=b"{}"
            )
        self.assertEqual(r.status_code, 504)
        self.assertEqual(r.json()["error"]["type"], "timeout_error")
        self.assertIsNone(r.json()["error"]["param"])
        self.assertEqual(gateway._admitted, 0)

    async def test_upstream_unreachable_returns_502(self) -> None:
        _install(_Raiser(lambda req: httpx.ConnectError("refused", request=req)))
        async with self._gw() as client:
            r = await client.post(
                "/v1/chat/completions", headers=AUTH, content=b"{}"
            )
        self.assertEqual(r.status_code, 502)
        self.assertEqual(r.json()["error"]["type"], "upstream_error")
        self.assertEqual(gateway._admitted, 0)

    async def test_upstream_error_status_is_passed_through(self) -> None:
        body = b'{"error": {"message": "bad", "type": "BadRequestError"}}'
        _install(_Recorder(status=400, content=body))
        async with self._gw() as client:
            r = await client.post(
                "/v1/chat/completions", headers=AUTH, content=b"{}"
            )
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.content, body)
        self.assertEqual(r.headers["content-type"], "application/json")
        self.assertEqual(gateway._admitted, 0)

    async def test_mid_stream_sse_failure_emits_error_and_done(self) -> None:
        up = _MidStreamFailure("text/event-stream")
        _install(up)
        async with self._gw() as client:
            r = await client.post(
                "/v1/chat/completions", headers=AUTH, content=b"{}"
            )
        self.assertEqual(r.status_code, 200)
        text = r.text
        self.assertTrue(text.startswith("A"))
        self.assertTrue(text.endswith("data: [DONE]\n\n"), text)
        frame = text[1:].split("\n\n")[0]
        self.assertTrue(frame.startswith("data: "), frame)
        error = json.loads(frame[len("data: "):])["error"]
        self.assertEqual(error["type"], "upstream_error")
        self.assertEqual(error["code"], 502)
        self.assertIsNone(error["param"])
        self.assertIn("upstream died", error["message"])
        self.assertEqual(gateway._admitted, 0)
        self.assertTrue(up.closed)

    async def test_mid_stream_non_sse_failure_truncates(self) -> None:
        up = _MidStreamFailure("application/json")
        _install(up)
        messages = []
        request_sent = False

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": b"{}", "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)

        scope = _scope(
            "/v1/chat/completions", [(b"authorization", b"Bearer test-key")]
        )
        with self.assertRaises(Exception):
            await gateway.app(scope, receive, send)

        start = [m for m in messages if m["type"] == "http.response.start"]
        self.assertEqual([m["status"] for m in start], [200])
        chunks = [m for m in messages if m["type"] == "http.response.body"]
        self.assertEqual(b"".join(m.get("body", b"") for m in chunks), b"A")
        # No terminating chunk: the client's transport must see a truncated body
        # rather than a silently complete one.
        self.assertTrue(all(m.get("more_body") for m in chunks), messages)
        self.assertEqual(gateway._admitted, 0)
        self.assertTrue(up.closed)

    async def test_reasoning_effort_high_is_mapped(self) -> None:
        recorder = _Recorder()
        _install(recorder)
        async with self._gw() as client:
            with self.assertLogs(gateway.log, level="INFO") as logs:
                r = await client.post(
                    "/v1/chat/completions",
                    headers=AUTH,
                    content=b'{"model": "m", "reasoning_effort": "high"}',
                )
            self.assertEqual(r.status_code, 200)
        self.assertTrue(
            any('mapped to "xhigh"' in line for line in logs.output), logs.output
        )
        sent = json.loads(recorder.last.content)
        self.assertEqual(sent, {"model": "m", "reasoning_effort": "xhigh"})

        untouched = [
            b'{"model": "m", "reasoning_effort": "xhigh"}',
            b'{"model": "m", "reasoning_effort": "medium"}',
            b'{"model": "m", "reasoning_effort": "low"}',
            b'{"model": "m", "reasoning_effort": "minimal"}',
            b'{"model": "m"}',
            b'{"model": "m", "nested": {"reasoning_effort": "high"}}',
            b'not json at all "reasoning_effort" "high"',
        ]
        for body in untouched:
            with self.subTest(body=body):
                recorder = _Recorder()
                _install(recorder)
                async with self._gw() as client:
                    r = await client.post(
                        "/v1/chat/completions", headers=AUTH, content=body
                    )
                    self.assertEqual(r.status_code, 200)
                self.assertEqual(recorder.last.content, body)

        # Only chat/completions is rewritten.
        recorder = _Recorder()
        _install(recorder)
        body = b'{"model": "m", "reasoning_effort": "high"}'
        async with self._gw() as client:
            r = await client.post("/v1/completions", headers=AUTH, content=body)
            self.assertEqual(r.status_code, 200)
        self.assertEqual(recorder.last.content, body)


def _run_gateway_import(env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    """Import gateway in a clean subprocess with only the given VLLM_* names set."""

    env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
    env.update(env_overrides)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import gateway;"
            "print(gateway.MAX_CONCURRENCY, gateway.MAX_QUEUE,"
            " gateway.QUEUE_TIMEOUT_SECONDS, gateway.READ_TIMEOUT_SECONDS,"
            " gateway.UPSTREAM_BASE == gateway.DEFAULT_UPSTREAM_BASE)",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


class ConfigurationTests(unittest.TestCase):
    def test_production_defaults(self) -> None:
        done = _run_gateway_import({"VLLM_API_KEY": "test-key"})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.split(), ["32", "32", "300.0", "3600.0", "True"])

    def test_invalid_configuration_fails_fast(self) -> None:
        cases = [
            ({"VLLM_GATEWAY_MAX_CONCURRENCY": "0"}, "MAX_CONCURRENCY must be"),
            ({"VLLM_GATEWAY_MAX_CONCURRENCY": "abc"}, "MAX_CONCURRENCY must be"),
            ({"VLLM_GATEWAY_MAX_QUEUE": "-1"}, "MAX_QUEUE must be"),
            ({"VLLM_GATEWAY_QUEUE_TIMEOUT_SECONDS": "0"}, "QUEUE_TIMEOUT"),
            ({"VLLM_GATEWAY_QUEUE_TIMEOUT_SECONDS": "nan"}, "QUEUE_TIMEOUT"),
            ({"VLLM_GATEWAY_READ_TIMEOUT_SECONDS": "-1"}, "READ_TIMEOUT"),
        ]
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                env = {"VLLM_API_KEY": "test-key"}
                env.update(overrides)
                done = _run_gateway_import(env)
                self.assertNotEqual(done.returncode, 0, done.stdout)
                self.assertIn(expected, done.stderr)

    def test_missing_api_key_fails_fast(self) -> None:
        # Also the regression guard for reading configuration from the process
        # environment only: the repository .env in the working directory holds a
        # real VLLM_API_KEY, so a gateway that still parsed it would start here.
        done = _run_gateway_import({})
        self.assertNotEqual(done.returncode, 0, done.stdout)
        self.assertIn("VLLM_API_KEY is required in the environment", done.stderr)


if __name__ == "__main__":
    unittest.main()
