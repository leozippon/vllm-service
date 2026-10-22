"""A throwaway repository tree for exercising the entry scripts as real processes.

These scripts decide whether a GPU box starts at all, so the tests run them as
real processes against a synthetic copy of ``scripts/`` with a synthetic model
profile and deployment. The engine itself is never launched: the runtime
interpreter ``.venv/bin/python`` is a recorder that writes its argv and
environment and exits.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RECORDER = """#!/usr/bin/env bash
printf '%s\\n' "$@" >"$ARGV_LOG"
printenv >"$ENV_LOG"
"""

PROFILE = "test-profile"
DEPLOYMENT = "test-deployment"
PORT = 65000

# Synthetic topology for instances: four GPUs on NUMA node 0, whose CPUs exist
# on any host that can run numactl.
GPU_NODES = [0, 0, 0, 0]
CPULISTS = {0: "0-1"}


def _assignments(values: dict[str, str | None]) -> list[str]:
    return [f"{k}={shlex.quote(v)}" for k, v in values.items() if v is not None]


class Repo:
    def __init__(self, root: Path) -> None:
        self.root = root
        shutil.copytree(REPO_ROOT / "scripts", root / "scripts")
        (root / "config" / "profiles").mkdir(parents=True)
        (root / "config" / "deployments").mkdir(parents=True)
        self.model = root / "model"
        self.model.mkdir()
        (self.model / "model.safetensors").write_bytes(b"\0" * 100)
        self.argv_log = root / "argv.log"
        self.env_log = root / "env.log"
        self.python = root / ".venv" / "bin" / "python"
        self.python.parent.mkdir(parents=True)
        self.python.write_text(RECORDER, encoding="utf-8")
        self.python.chmod(0o755)
        self.sysfs = write_topology(root, GPU_NODES, CPULISTS)
        self.write_selector(DEPLOYMENT)
        self.write_profile()
        self.write_deployment()

    def write_selector(self, value: str) -> None:
        (self.root / "config" / "deployments" / "active").write_text(
            f"{value}\n", encoding="utf-8"
        )

    def write_profile(
        self,
        *,
        name: str = PROFILE,
        args: str | None = "(--enable-prefix-caching)",
        **overrides: str | None,
    ) -> None:
        """Write a profile; an override of None leaves the name out entirely."""

        values: dict[str, str | None] = {
            "VLLM_MODEL_PATH": str(self.model),
            "VLLM_SERVED_MODEL_NAME": "test-model",
            "VLLM_TASK": "generate",
            "VLLM_TENSOR_PARALLEL_SIZE": "1",
            "VLLM_MAX_MODEL_LEN": "4096",
            "VLLM_RELEASE_DATE": "2026-01",
            "VLLM_KNOWLEDGE_CUTOFF": "",
            "VLLM_NOTE": "a test model",
        }
        values.update(overrides)
        lines = _assignments(values)
        if args is not None:
            lines.append(f"VLLM_ARGS={args}")
        (self.root / "config" / "profiles" / f"{name}.conf").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def write_deployment(
        self,
        *,
        deployment: str = DEPLOYMENT,
        launch_args: str | None = None,
        **overrides: str | None,
    ) -> None:
        """Write a deployment; an override of None leaves the name out entirely."""

        values: dict[str, str | None] = {
            "VLLM_PROFILE": PROFILE,
            "VLLM_PORT": str(PORT),
            "VLLM_GPUS": "0",
            "VLLM_CPU_AFFINITY": "0",
            "VLLM_NUMA_NODES": "0",
        }
        values.update(overrides)
        lines = _assignments(values)
        if launch_args is not None:
            lines.append(f"VLLM_LAUNCH_ARGS={launch_args}")
        (self.root / "config" / "deployments" / f"{deployment}.conf").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def run(
        self, script: str, *args: str, **extra_env: str
    ) -> subprocess.CompletedProcess:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
            "VLLM_API_KEY": "test-key",
            "ARGV_LOG": str(self.argv_log),
            "ENV_LOG": str(self.env_log),
            "VLLM_SYSFS_ROOT": str(self.sysfs),
            # Never let a test sweep the real /dev/shm.
            "VLLM_OFFLOAD_DIR": str(self.root),
        }
        env.update(extra_env)
        return subprocess.run(
            [self.root / "scripts" / script, *args],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def argv(self) -> list[str]:
        return self.argv_log.read_text(encoding="utf-8").splitlines()

    def recorded_env(self) -> dict[str, str]:
        pairs = (
            line.partition("=")
            for line in self.env_log.read_text(encoding="utf-8").splitlines()
        )
        return {name: value for name, _, value in pairs}


def write_topology(root: Path, gpu_nodes: list[int], cpulists: dict[int, str]) -> Path:
    """A synthetic sysfs: NVIDIA devices in PCI bus order, and their NUMA nodes.

    ``gpu_nodes[i]`` is the NUMA node of the GPU that CUDA would number ``i``.
    """

    sysfs = root / "sysfs"
    shutil.rmtree(sysfs, ignore_errors=True)
    drivers = sysfs / "bus" / "pci" / "drivers" / "nvidia"
    drivers.mkdir(parents=True)
    # The driver directory also holds bind/unbind/module; only devices count.
    (drivers / "bind").touch()
    for index, node in enumerate(gpu_nodes):
        device = drivers / f"0000:{index:02x}:00.0"
        device.mkdir()
        (device / "numa_node").write_text(f"{node}\n", encoding="utf-8")
    for node, cpulist in cpulists.items():
        directory = sysfs / "devices" / "system" / "node" / f"node{node}"
        directory.mkdir(parents=True)
        (directory / "cpulist").write_text(f"{cpulist}\n", encoding="utf-8")
    return sysfs
