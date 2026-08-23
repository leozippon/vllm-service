"""Negative paths and configuration authority in the deployment entry scripts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _repo import PORT, Repo, write_topology


class EntryScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Repo(Path(self._tmp.name))

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
                repo = Repo(Path(raw))
                mutate(repo)
                result = repo.run("run-vllm")
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(message, result.stderr)
                self.assertFalse(repo.argv_log.exists())

    def test_deployment_override_resolves_without_the_selector(self) -> None:
        """A study run or a per-instance unit selects a profile from the environment."""
        self.repo.write_profile(
            deployment="candidate", VLLM_SERVED_MODEL_NAME="candidate-model"
        )
        (self.repo.root / "config" / "deployments" / "active").unlink()

        result = self.repo.run("run-vllm", VLLM_DEPLOYMENT="candidate")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = self.repo.argv()
        self.assertEqual(
            argv[argv.index("--served-model-name") + 1], "candidate-model"
        )

        # The override is validated exactly like the selector's own value.
        result = self.repo.run("run-vllm", VLLM_DEPLOYMENT="../escape")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("deployment ID is empty or invalid", result.stderr)

    def test_missing_api_key_fails_fast(self) -> None:
        result = self.repo.run("run-vllm", VLLM_API_KEY="")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("VLLM_API_KEY is required", result.stderr)
        self.assertFalse(self.repo.argv_log.exists())

    def test_api_key_never_reaches_the_command_line(self) -> None:
        """/proc/<pid>/cmdline is world-readable on this shared host."""
        result = self.repo.run("run-vllm")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = self.repo.argv()
        self.assertEqual(argv[argv.index("--served-model-name") + 1], "test-model")
        for token in argv:
            self.assertNotIn("test-key", token, argv)

    def test_inherited_environment_does_not_satisfy_a_profile_requirement(self) -> None:
        """B11: the profile is the authority; an inherited value must not stand in."""
        self.repo.write_profile(VLLM_SERVED_MODEL_NAME=None)
        result = self.repo.run("run-vllm", VLLM_SERVED_MODEL_NAME="inherited")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("VLLM_SERVED_MODEL_NAME is required in", result.stderr)
        self.assertFalse(self.repo.argv_log.exists())

    def test_gateway_shares_the_same_preamble(self) -> None:
        self.repo.write_profile(VLLM_PYTHON=None)
        result = self.repo.run("run-gateway", VLLM_PYTHON="/bin/sh")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("VLLM_PYTHON is required in", result.stderr)

        self.repo.write_profile()
        result = self.repo.run("run-gateway")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            self.repo.argv()[:3], ["-m", "uvicorn", "gateway:app"]
        )

    def test_placement_is_derived_when_the_profile_does_not_state_it(self) -> None:
        """A profile must not carry another machine's core numbering."""
        sysfs = write_topology(self.repo.root, [0], {0: "0-3"})
        self.repo.write_profile(
            VLLM_CPU_AFFINITY=None, VLLM_NUMA_NODES=None, VLLM_CPU_CORES="2"
        )
        result = self.repo.run("run-vllm", VLLM_SYSFS_ROOT=str(sysfs))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("placement node=0 cpus=0-1 (gpus 0)", result.stdout)

        # The count is policy, not topology, so it is never guessed.
        self.repo.write_profile(VLLM_CPU_AFFINITY=None, VLLM_NUMA_NODES=None)
        result = self.repo.run("run-vllm", VLLM_SYSFS_ROOT=str(sysfs))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("VLLM_CPU_CORES is required in", result.stderr)

    def test_a_gpu_outside_the_host_allowlist_is_refused(self) -> None:
        """On a shared host the profile must not be the only thing keeping us off
        another tenant's GPU."""
        self.repo.write_profile(VLLM_GPUS="3")
        result = self.repo.run("run-vllm", VLLM_HOST_GPUS="6,7")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("GPU 3 is not in VLLM_HOST_GPUS (6,7)", result.stderr)
        self.assertFalse(self.repo.argv_log.exists())

        # Membership is by whole index, not by substring.
        self.repo.write_profile(VLLM_GPUS="1")
        result = self.repo.run("run-vllm", VLLM_HOST_GPUS="10,11")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("GPU 1 is not in VLLM_HOST_GPUS", result.stderr)

        result = self.repo.run("run-vllm", VLLM_HOST_GPUS="0,1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_one_declared_port_reaches_the_engine_and_the_gateway(self) -> None:
        """The profile declares the engine port; everything else follows from it."""
        result = self.repo.run("run-vllm")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = self.repo.argv()
        self.assertEqual(argv[argv.index("--port") + 1], str(PORT))

        result = self.repo.run("run-gateway")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = self.repo.argv()
        self.assertEqual(argv[argv.index("--port") + 1], str(PORT + 1))
        self.assertEqual(
            self.repo.recorded_env()["VLLM_GATEWAY_UPSTREAM_URL"],
            f"http://127.0.0.1:{PORT}",
        )

        # .env still redirects the upstream, which is what a remote engine needs.
        redirected = "http://elsewhere.invalid:9000"
        result = self.repo.run("run-gateway", VLLM_GATEWAY_UPSTREAM_URL=redirected)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            self.repo.recorded_env()["VLLM_GATEWAY_UPSTREAM_URL"], redirected
        )

    def test_unusable_port_fails_before_the_engine_loads(self) -> None:
        """A four-minute load must not end in a gate polling a port nobody bound."""
        for value in ("", "8010x", "65535"):
            with self.subTest(port=value):
                self.repo.write_profile(VLLM_PORT=value or None)
                for script in ("run-vllm", "run-gateway"):
                    result = self.repo.run(script)
                    self.assertEqual(
                        result.returncode, 1, result.stdout + result.stderr
                    )
                    self.assertIn("VLLM_PORT must be a TCP port", result.stderr)
                self.assertFalse(self.repo.argv_log.exists())


if __name__ == "__main__":
    unittest.main()
