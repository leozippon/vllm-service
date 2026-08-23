"""Deriving CPU and NUMA placement from the GPU set, and refusing to guess.

The derivation replaces a hand-written placement that was silently wrong on any
machine but this one, so the acceptance test is exact equality with the value it
replaced: this host's GPUs 6 and 7 sit on NUMA node 1, whose CPU pool starts at
48, and the profile said node 1 / 48-63.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from _repo import REPO_ROOT, write_topology

RESOLVE = REPO_ROOT / "scripts" / "resolve-placement"

HOST_GPU_NODES = [0, 0, 0, 0, 1, 1, 1, 1]
HOST_CPULISTS = {0: "0-47,96-143", 1: "48-95,144-191"}


class ResolvePlacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def resolve(
        self,
        gpus: str,
        cores: str,
        *,
        gpu_nodes: list[int] | None = None,
        cpulists: dict[int, str] | None = None,
        device_order: str = "PCI_BUS_ID",
    ) -> subprocess.CompletedProcess:
        sysfs = write_topology(
            self.root,
            HOST_GPU_NODES if gpu_nodes is None else gpu_nodes,
            HOST_CPULISTS if cpulists is None else cpulists,
        )
        return subprocess.run(
            [RESOLVE, gpus, cores],
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "VLLM_SYSFS_ROOT": str(sysfs),
                "CUDA_DEVICE_ORDER": device_order,
            },
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_derivation_reproduces_the_hand_written_placement(self) -> None:
        result = self.resolve("6,7", "16")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "1 48-63")

    def test_a_split_pool_comes_back_as_ranges(self) -> None:
        """numactl takes either form; a range is what an operator would write."""
        result = self.resolve("0", "4", gpu_nodes=[0], cpulists={0: "0-1,4-7"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "0 0-1,4-5")

    def test_untrustworthy_topology_fails_instead_of_guessing(self) -> None:
        cases = {
            "span NUMA nodes 0 and 1": dict(gpus="3,4", cores="16"),
            "reports no NUMA affinity": dict(gpus="0", cores="16", gpu_nodes=[-1]),
            "fewer than the 16 requested": dict(
                gpus="6,7", cores="16", cpulists={1: "48-51"}
            ),
            "no such GPU: 8": dict(gpus="8", cores="16"),
            "CUDA_DEVICE_ORDER must be PCI_BUS_ID": dict(
                gpus="6,7", cores="16", device_order=""
            ),
        }
        for message, kwargs in cases.items():
            with self.subTest(message=message):
                result = self.resolve(**kwargs)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(message, result.stderr)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
