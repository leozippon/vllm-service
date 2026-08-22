"""Negative paths and configuration authority in the deployment entry scripts.

These scripts decide whether the GPU box starts at all, so the tests run them as
real processes against a throwaway repository tree. The engine itself is never
launched: the profile's VLLM_PYTHON is a recorder that writes its argv and exits.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RECORDER = """#!/usr/bin/env bash
printf '%s\\n' "$@" >"$ARGV_LOG"
"""


class _Repo:
    """A throwaway copy of scripts/ with a synthetic deployment profile."""

    def __init__(self, root: Path) -> None:
        self.root = root
        shutil.copytree(REPO_ROOT / "scripts", root / "scripts")
        (root / "config" / "deployments").mkdir(parents=True)
        (root / "model").mkdir()
        self.argv_log = root / "argv.log"
        self.python = root / "recorder"
        self.python.write_text(RECORDER, encoding="utf-8")
        self.python.chmod(0o755)
        self.write_selector("test-deployment")
        self.write_profile()

    def write_selector(self, value: str) -> None:
        (self.root / "config" / "deployments" / "active").write_text(
            f"{value}\n", encoding="utf-8"
        )

    def write_profile(self, *, omit: str = "") -> None:
        values = {
            "VLLM_PYTHON": str(self.python),
            "VLLM_MODEL_PATH": str(self.root / "model"),
            "VLLM_SERVED_MODEL_NAME": "test-model",
            "VLLM_GPUS": "0",
            "VLLM_CPU_AFFINITY": "0",
            "VLLM_NUMA_NODES": "0",
        }
        values.pop(omit, None)
        lines = [f"{k}={v}" for k, v in values.items()]
        lines.append('VLLM_RUNTIME_ENV=("TEST_MARKER=1")')
        lines.append("VLLM_ARGS=(--host 127.0.0.1 --port 65000)")
        (self.root / "config" / "deployments" / "test-deployment.conf").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def run(self, script: str, **extra_env: str) -> subprocess.CompletedProcess:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
            "VLLM_API_KEY": "test-key",
            "ARGV_LOG": str(self.argv_log),
            # Never let a test sweep the real /dev/shm.
            "VLLM_OFFLOAD_DIR": str(self.root),
        }
        env.update(extra_env)
        return subprocess.run(
            [self.root / "scripts" / script],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )


class EntryScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = _Repo(Path(self._tmp.name))

    def test_unusable_deployment_selection_fails_fast(self) -> None:
        cases = {
            "selector is not readable": lambda repo: (
                repo.root / "config" / "deployments" / "active"
            ).unlink(),
            "deployment ID is empty or invalid": lambda repo: repo.write_selector(
                "../escape"
            ),
            "profile is not readable": lambda repo: repo.write_selector("absent"),
        }
        for message, mutate in cases.items():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as raw:
                repo = _Repo(Path(raw))
                mutate(repo)
                result = repo.run("run-vllm")
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(message, result.stderr)
                self.assertFalse(repo.argv_log.exists())

    def test_missing_api_key_fails_fast(self) -> None:
        result = self.repo.run("run-vllm", VLLM_API_KEY="")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("VLLM_API_KEY is required", result.stderr)
        self.assertFalse(self.repo.argv_log.exists())

    def test_api_key_never_reaches_the_command_line(self) -> None:
        """/proc/<pid>/cmdline is world-readable on this shared host."""
        result = self.repo.run("run-vllm")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = self.repo.argv_log.read_text(encoding="utf-8").split("\n")
        self.assertEqual(argv[argv.index("--served-model-name") + 1], "test-model")
        for token in argv:
            self.assertNotIn("test-key", token, argv)

    def test_inherited_environment_does_not_satisfy_a_profile_requirement(self) -> None:
        """B11: the profile is the authority; an inherited value must not stand in."""
        self.repo.write_profile(omit="VLLM_SERVED_MODEL_NAME")
        result = self.repo.run("run-vllm", VLLM_SERVED_MODEL_NAME="inherited")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("VLLM_SERVED_MODEL_NAME is required in", result.stderr)
        self.assertFalse(self.repo.argv_log.exists())

    def test_gateway_shares_the_same_preamble(self) -> None:
        self.repo.write_profile(omit="VLLM_PYTHON")
        result = self.repo.run("run-gateway", VLLM_PYTHON="/bin/sh")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("VLLM_PYTHON is required in", result.stderr)

        self.repo.write_profile()
        result = self.repo.run("run-gateway")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            self.repo.argv_log.read_text(encoding="utf-8").split("\n")[:4],
            ["-m", "uvicorn", "gateway:app", "--host"],
        )


if __name__ == "__main__":
    unittest.main()
