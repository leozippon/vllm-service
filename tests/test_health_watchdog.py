"""Focused regression test for watchdog restart failure handling."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

WATCHDOG = Path(__file__).resolve().parent.parent / "scripts" / "health-watchdog"


class WatchdogTests(unittest.TestCase):
    def test_restart_failure_remains_failed_until_restart_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            bin_dir = tmp / "bin"
            bin_dir.mkdir()
            systemctl_log = tmp / "systemctl.log"

            curl = bin_dir / "curl"
            curl.write_text(
                "#!/usr/bin/env bash\n"
                "url=${!#}\n"
                "[[ \"$url\" != \"$FAIL_URL\" ]]\n",
                encoding="utf-8",
            )
            systemctl = bin_dir / "systemctl"
            systemctl.write_text(
                "#!/usr/bin/env bash\n"
                "case ${2:-} in\n"
                "  show) printf 'active\\n' ;;\n"
                "  restart)\n"
                "    printf '%s\\n' \"${3:-}\" >>\"$SYSTEMCTL_LOG\"\n"
                "    [[ ${RESTART_OK:-0} == 1 ]]\n"
                "    ;;\n"
                "  *) exit 2 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            curl.chmod(0o755)
            systemctl.chmod(0o755)

            backend_url = "http://backend.invalid/health"
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{bin_dir}:{env['PATH']}",
                    "XDG_RUNTIME_DIR": str(tmp / "runtime"),
                    "FAIL_URL": backend_url,
                    "SYSTEMCTL_LOG": str(systemctl_log),
                    "VLLM_WATCHDOG_UPSTREAM_URL": backend_url,
                    "VLLM_WATCHDOG_GATEWAY_URL": "http://gateway.invalid/health/live",
                    "VLLM_WATCHDOG_BACKEND_UNIT": "test-backend.service",
                    "VLLM_WATCHDOG_GATEWAY_UNIT": "test-gateway.service",
                    "VLLM_WATCHDOG_BACKEND_THRESHOLD": "1",
                    "VLLM_WATCHDOG_GATEWAY_THRESHOLD": "1",
                }
            )

            failed = subprocess.run(
                [WATCHDOG], env=env, check=False, capture_output=True, text=True
            )
            state_dir = tmp / "runtime" / "vllm-watchdog"
            self.assertEqual(failed.returncode, 1, failed.stdout + failed.stderr)
            self.assertEqual((state_dir / "backend.fails").read_text().strip(), "1")
            self.assertEqual((state_dir / "gateway.fails").read_text().strip(), "0")
            self.assertEqual(systemctl_log.read_text().strip(), "test-backend.service")

            env["RESTART_OK"] = "1"
            recovered = subprocess.run(
                [WATCHDOG], env=env, check=False, capture_output=True, text=True
            )
            self.assertEqual(recovered.returncode, 0, recovered.stdout + recovered.stderr)
            self.assertEqual((state_dir / "backend.fails").read_text().strip(), "0")


if __name__ == "__main__":
    unittest.main()
