"""Watchdog state machine: what it restarts, what it skips, and what it reports.

Exit status 0 must mean the system was healthy this cycle. A restart is an
intervention, not health, so a cycle that restarted something exits non-zero.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

WATCHDOG = Path(__file__).resolve().parent.parent / "scripts" / "health-watchdog"

BACKEND_URL = "http://backend.invalid/health"
GATEWAY_URL = "http://gateway.invalid/health/live"

CURL = """#!/usr/bin/env bash
url=${!#}
printf '%s\\n' "$url" >>"$CURL_LOG"
[[ "$url" != "$FAIL_URL" ]]
"""

SYSTEMCTL = """#!/usr/bin/env bash
case ${2:-} in
  show) printf '%s\\n' "${SHOW_STATE:-active}" ;;
  restart)
    printf '%s\\n' "$*" >>"$SYSTEMCTL_LOG"
    [[ ${RESTART_OK:-0} == 1 ]]
    ;;
  *) exit 2 ;;
esac
"""


class WatchdogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        for name, body in (("curl", CURL), ("systemctl", SYSTEMCTL)):
            stub = bin_dir / name
            stub.write_text(body, encoding="utf-8")
            stub.chmod(0o755)
        self.curl_log = self.tmp / "curl.log"
        self.systemctl_log = self.tmp / "systemctl.log"
        self.state_dir = self.tmp / "runtime" / "vllm-watchdog"
        self.env = {
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "XDG_RUNTIME_DIR": str(self.tmp / "runtime"),
            "FAIL_URL": BACKEND_URL,
            "CURL_LOG": str(self.curl_log),
            "SYSTEMCTL_LOG": str(self.systemctl_log),
            "VLLM_WATCHDOG_UPSTREAM_URL": BACKEND_URL,
            "VLLM_WATCHDOG_GATEWAY_URL": GATEWAY_URL,
            "VLLM_WATCHDOG_BACKEND_UNIT": "test-backend.service",
            "VLLM_WATCHDOG_GATEWAY_UNIT": "test-gateway.service",
            "VLLM_WATCHDOG_BACKEND_THRESHOLD": "1",
            "VLLM_WATCHDOG_GATEWAY_THRESHOLD": "1",
        }

    def run_watchdog(self, **overrides: str) -> subprocess.CompletedProcess:
        env = dict(self.env)
        env.update(overrides)
        return subprocess.run(
            [WATCHDOG], env=env, check=False, capture_output=True, text=True
        )

    def counter(self, name: str) -> str:
        return (self.state_dir / f"{name}.fails").read_text(encoding="utf-8").strip()

    def test_healthy_cycle_exits_zero(self) -> None:
        result = self.run_watchdog(FAIL_URL="")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.systemctl_log.exists(), result.stdout)
        self.assertEqual(self.counter("backend"), "0")
        self.assertEqual(self.counter("gateway"), "0")

    def test_restart_is_an_intervention_not_health(self) -> None:
        failed = self.run_watchdog()
        self.assertEqual(failed.returncode, 1, failed.stdout + failed.stderr)
        self.assertEqual(self.counter("backend"), "1")
        self.assertEqual(self.counter("gateway"), "0")
        # A failed restart must leave the counter standing.
        self.assertEqual(
            self.systemctl_log.read_text(encoding="utf-8").split(),
            ["--user", "restart", "--no-block", "test-backend.service"],
        )

        restarted = self.run_watchdog(RESTART_OK="1")
        self.assertEqual(restarted.returncode, 1, restarted.stdout + restarted.stderr)
        self.assertEqual(self.counter("backend"), "0")

    def test_threshold_accumulates_without_restarting(self) -> None:
        for expected in ("1", "2"):
            result = self.run_watchdog(VLLM_WATCHDOG_BACKEND_THRESHOLD="3")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertEqual(self.counter("backend"), expected)
            self.assertFalse(self.systemctl_log.exists(), result.stdout)

    def test_transitioning_units_are_skipped(self) -> None:
        for state in ("activating", "deactivating"):
            with self.subTest(state=state):
                result = self.run_watchdog(SHOW_STATE=state)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"backend skip ({state})", result.stdout)
                self.assertFalse(self.curl_log.exists(), result.stdout)
                self.assertFalse(self.systemctl_log.exists(), result.stdout)
                self.assertFalse((self.state_dir / "backend.fails").exists())

    def test_corrupt_counter_is_treated_as_zero(self) -> None:
        self.state_dir.mkdir(parents=True)
        (self.state_dir / "backend.fails").write_text("abc\n", encoding="utf-8")
        result = self.run_watchdog(VLLM_WATCHDOG_BACKEND_THRESHOLD="3")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(self.counter("backend"), "1")
        self.assertFalse(self.systemctl_log.exists(), result.stdout)

    def test_invalid_threshold_exits_two(self) -> None:
        result = self.run_watchdog(VLLM_WATCHDOG_BACKEND_THRESHOLD="0")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("must be a positive integer", result.stderr)


if __name__ == "__main__":
    unittest.main()
