"""The ExecStartPost readiness gate: when it releases the unit and when it fails."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

WAIT_READY = Path(__file__).resolve().parent.parent / "scripts" / "wait-ready"

CURL = """#!/usr/bin/env bash
url=${!#}
case "$url" in
  */health) [[ ${HEALTH_OK:-1} == 1 ]] ;;
  */v1/models) printf '%s' "${AUTH_STATUS:-401}" ;;
  *) exit 1 ;;
esac
"""


class WaitReadyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        bin_dir = Path(self._tmp.name) / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "curl"
        stub.write_text(CURL, encoding="utf-8")
        stub.chmod(0o755)
        self.env = {
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "VLLM_READY_URL": "http://upstream.invalid/health",
            "VLLM_READY_INTERVAL_SECONDS": "0.1",
        }

    def run_gate(self, **overrides: str) -> subprocess.CompletedProcess:
        env = dict(self.env)
        env.update(overrides)
        return subprocess.run(
            [WAIT_READY], env=env, check=False, capture_output=True, text=True,
            timeout=30,
        )

    def test_serving_and_authenticated_upstream_releases_the_unit(self) -> None:
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("unauthenticated requests are rejected", result.stdout)

    def test_unauthenticated_upstream_fails_the_start(self) -> None:
        result = self.run_gate(AUTH_STATUS="200")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("returned 200, expected 401", result.stderr)

    def test_dead_main_process_fails_instead_of_waiting_out_the_timeout(self) -> None:
        dead = subprocess.Popen(["true"])
        dead.wait()
        result = self.run_gate(HEALTH_OK="0", MAINPID=str(dead.pid))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("exited before becoming ready", result.stderr)


if __name__ == "__main__":
    unittest.main()
