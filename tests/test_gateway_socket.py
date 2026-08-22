"""End-to-end gateway behaviour over a real socket.

``httpx.ASGITransport`` collects ASGI messages in process and never exercises
uvicorn's chunked framing, which is exactly where a mid-stream upstream failure
becomes visible (or invisible) to a client. One real-server test covers it; the
upstream stays a stub, because only the client-to-gateway socket needs to be
real.
"""

from __future__ import annotations

import os
import socket
import threading
import time
import unittest

os.environ.setdefault("VLLM_API_KEY", "test-key")

import httpx
import uvicorn

import gateway

AUTH = {"Authorization": "Bearer test-key"}


class _Upstream(httpx.AsyncBaseTransport):
    """Streams one SSE frame, then either finishes cleanly or dies."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        fail = request.headers.get("x-test-fail") == "1"

        class _Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"id": "1"}\n\n'
                if fail:
                    raise RuntimeError("upstream died")
                yield b"data: [DONE]\n\n"

            async def aclose(self) -> None:
                return None

        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_Body(),
            request=request,
        )


class SocketTests(unittest.TestCase):
    def test_streaming_over_a_real_server(self) -> None:
        previous = gateway.client
        gateway.client = httpx.AsyncClient(
            transport=_Upstream(), base_url="http://upstream.invalid"
        )
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(
                gateway.app,
                host="127.0.0.1",
                port=port,
                lifespan="off",
                log_config=None,
                access_log=False,
            )
        )
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [listener]}, daemon=True
        )
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(server.started, "uvicorn did not start")

            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as c:
                bad = c.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer wrong"},
                    content=b"{}",
                )
                self.assertEqual(bad.status_code, 401)
                self.assertEqual(bad.json()["error"]["type"], "auth_error")

                # A complete stream: clean chunked framing, terminated by [DONE].
                ok = c.post("/v1/chat/completions", headers=AUTH, content=b"{}")
                self.assertEqual(ok.status_code, 200)
                self.assertEqual(ok.text, 'data: {"id": "1"}\n\ndata: [DONE]\n\n')

                # A mid-stream upstream failure must reach the client as an
                # OpenAI-shaped error frame plus [DONE], with the body framing
                # still closed cleanly -- reading it must not raise
                # RemoteProtocolError.
                broken = c.post(
                    "/v1/chat/completions",
                    headers={**AUTH, "x-test-fail": "1"},
                    content=b"{}",
                )
                self.assertEqual(broken.status_code, 200)
                self.assertTrue(broken.text.startswith('data: {"id": "1"}\n\n'))
                self.assertTrue(broken.text.endswith("data: [DONE]\n\n"), broken.text)
                self.assertIn('"type": "upstream_error"', broken.text)

            self.assertEqual(gateway._admitted, 0)
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive(), "uvicorn did not stop")
            gateway.client = previous


if __name__ == "__main__":
    unittest.main()
