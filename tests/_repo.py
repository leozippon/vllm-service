"""A throwaway repository tree for exercising the entry scripts as real processes.

These scripts decide whether the GPU box starts at all, so the tests run them as
real processes against a synthetic copy of ``scripts/`` with a synthetic
deployment profile. The engine itself is never launched: the profile's
VLLM_PYTHON is a recorder that writes its argv and environment and exits.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RECORDER = """#!/usr/bin/env bash
printf '%s\\n' "$@" >"$ARGV_LOG"
printenv >"$ENV_LOG"
"""

DEPLOYMENT = "test-deployment"
PORT = 65000


class Repo:
    def __init__(self, root: Path) -> None:
        self.root = root
        shutil.copytree(REPO_ROOT / "scripts", root / "scripts")
        (root / "config" / "deployments").mkdir(parents=True)
        (root / "model").mkdir()
        self.argv_log = root / "argv.log"
        self.env_log = root / "env.log"
        self.python = root / "recorder"
        self.python.write_text(RECORDER, encoding="utf-8")
        self.python.chmod(0o755)
        self.write_selector(DEPLOYMENT)
        self.write_profile()

    def write_selector(self, value: str) -> None:
        (self.root / "config" / "deployments" / "active").write_text(
            f"{value}\n", encoding="utf-8"
        )

    def write_profile(
        self, *, deployment: str = DEPLOYMENT, omit: str = "", **overrides: str
    ) -> None:
        values = {
            "VLLM_PYTHON": str(self.python),
            "VLLM_MODEL_PATH": str(self.root / "model"),
            "VLLM_SERVED_MODEL_NAME": "test-model",
            "VLLM_PORT": str(PORT),
            "VLLM_GPUS": "0",
            "VLLM_CPU_AFFINITY": "0",
            "VLLM_NUMA_NODES": "0",
        }
        values.update(overrides)
        values.pop(omit, None)
        lines = [f"{k}={v}" for k, v in values.items()]
        lines.append('VLLM_RUNTIME_ENV=("TEST_MARKER=1")')
        lines.append("VLLM_ARGS=(--host 127.0.0.1)")
        (self.root / "config" / "deployments" / f"{deployment}.conf").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def run(self, script: str, **extra_env: str) -> subprocess.CompletedProcess:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
            "VLLM_API_KEY": "test-key",
            "ARGV_LOG": str(self.argv_log),
            "ENV_LOG": str(self.env_log),
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
            timeout=60,
        )

    def argv(self) -> list[str]:
        return self.argv_log.read_text(encoding="utf-8").split("\n")

    def recorded_env(self) -> dict[str, str]:
        pairs = (
            line.partition("=")
            for line in self.env_log.read_text(encoding="utf-8").splitlines()
        )
        return {name: value for name, _, value in pairs}
