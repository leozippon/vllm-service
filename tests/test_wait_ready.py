"""The ExecStartPost readiness gate: when it releases the unit and when it fails."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from _repo import PORT, Repo

CURL = """#!/usr/bin/env bash
url=${!#}
printf '%s\\n' "$url" >>"$CURL_LOG"
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
        self.repo = Repo(Path(self._tmp.name))
        bin_dir = self.repo.root / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "curl"
        stub.write_text(CURL, encoding="utf-8")
        stub.chmod(0o755)
        self.curl_log = self.repo.root / "curl.log"
        self.env = {
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "CURL_LOG": str(self.curl_log),
            "VLLM_READY_INTERVAL_SECONDS": "0.1",
        }

    def run_gate(self, **overrides: str) -> subprocess.CompletedProcess:
        env = dict(self.env)
        env.update(overrides)
        return self.repo.run("wait-ready", **env)

    def probed(self) -> list[str]:
        return self.curl_log.read_text(encoding="utf-8").split()

    def test_serving_and_authenticated_upstream_releases_the_unit(self) -> None:
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("unauthenticated requests are rejected", result.stdout)
        # The gate probes the port the profile told the engine to bind, so a
        # port change cannot leave it polling a dead endpoint for the whole
        # start timeout after a full model load.
        self.assertEqual(
            self.probed(),
            [f"http://127.0.0.1:{PORT}/health", f"http://127.0.0.1:{PORT}/v1/models"],
        )

    def test_explicit_url_overrides_the_derived_one(self) -> None:
        result = self.run_gate(VLLM_READY_URL="http://upstream.invalid/health")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.probed()[0], "http://upstream.invalid/health")

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

    def test_unreadable_profile_fails_the_start(self) -> None:
        """Nothing can say where the engine should answer, so the unit must fail."""
        (self.repo.root / "config" / "deployments" / "active").unlink()
        result = self.run_gate()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("selector is not readable", result.stderr)
        self.assertFalse(self.curl_log.exists())


if __name__ == "__main__":
    unittest.main()
