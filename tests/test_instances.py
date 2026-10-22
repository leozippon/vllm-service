"""The instance launcher: argument parsing, profile loading, and what reaches vLLM."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _repo import Repo

BASE = ("--profile", "test-profile", "--gpus", "0", "--port", "65100")


class InstanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Repo(Path(self._tmp.name))

    def launch(self, *args: str, **env: str) -> list[str]:
        result = self.repo.run("run-vllm", *args, **env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return self.repo.argv()

    def assert_refused(self, message: str, *args: str) -> None:
        result = self.repo.run("run-vllm", *args)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)
        self.assertFalse(self.repo.argv_log.exists())

    def test_profile_defaults_reach_the_engine_on_loopback(self) -> None:
        self.assertEqual(
            self.launch(*BASE),
            [
                "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(self.repo.model),
                "--served-model-name", "test-model",
                "--host", "127.0.0.1",
                "--port", "65100",
                "--runner", "generate",
                "--tensor-parallel-size", "1",
                "--max-model-len", "4096",
                "--enable-prefix-caching",
            ],
        )
        env = self.repo.recorded_env()
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_options_override_defaults_and_extra_arguments_come_last(self) -> None:
        argv = self.launch(
            "--profile=test-profile", "--gpus", "2,3", "--port", "65100",
            "--tensor-parallel-size", "2", "--max-model-len=8192",
            "--gpu-memory-utilization", "0.5",
            "--extra-arg", "--max-num-seqs", "--extra-arg", "8",
            "--extra-arg=--default-chat-template-kwargs={\"enable_thinking\": true}",
        )
        self.assertEqual(
            argv[argv.index("--tensor-parallel-size"):],
            [
                "--tensor-parallel-size", "2",
                "--max-model-len", "8192",
                "--gpu-memory-utilization", "0.5",
                "--enable-prefix-caching",
                "--max-num-seqs", "8",
                "--default-chat-template-kwargs={\"enable_thinking\": true}",
            ],
        )
        self.assertEqual(self.repo.recorded_env()["CUDA_VISIBLE_DEVICES"], "2,3")

    def test_embedding_profiles_run_the_pooling_runner(self) -> None:
        self.repo.write_profile(VLLM_TASK="embed", args="()")
        argv = self.launch(*BASE)
        self.assertEqual(
            argv[argv.index("--runner"):],
            [
                "--runner", "pooling", "--convert", "embed",
                "--tensor-parallel-size", "1", "--max-model-len", "4096",
            ],
        )

    def test_an_instance_needs_no_key_but_says_so(self) -> None:
        result = self.repo.run("run-vllm", *BASE, VLLM_API_KEY="")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("accepts unauthenticated requests", result.stderr)

    def test_an_instance_binds_its_gpus_whole_node(self) -> None:
        """Placement follows the GPUs; deployment policy never arrives from the
        caller's environment."""
        result = self.repo.run(
            "run-vllm", *BASE,
            VLLM_CPU_CORES="1", VLLM_NUMA_NODES="1", VLLM_CPU_AFFINITY="5",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("placement node=0 cpus=0-1 (gpus 0)", result.stdout)

    def test_unusable_command_lines_fail_before_launch(self) -> None:
        cases = [
            ("--profile, --gpus and --port are required", BASE[:4]),
            ("unknown option: --model", (*BASE, "--model", "x")),
            ("unexpected argument: serve", ("serve", *BASE)),
            ("--port needs a value", (*BASE[:4], "--port")),
            ("--gpus must be comma-separated GPU indices", (*BASE[:2], "--gpus", "0,a", *BASE[4:])),
            ("--port must be a TCP port", (*BASE[:4], "--port", "70000")),
            ("--tensor-parallel-size must be a positive integer", (*BASE, "--tensor-parallel-size", "0")),
            ("--max-model-len must be a positive integer", (*BASE, "--max-model-len", "32k")),
            ("--gpu-memory-utilization must be a fraction", (*BASE, "--gpu-memory-utilization", "1.5")),
            ("names 2 GPU(s) but the tensor-parallel size is 1", (*BASE[:2], "--gpus", "0,1", *BASE[4:])),
            ("names a GPU twice", (*BASE[:2], "--gpus", "1,1", *BASE[4:], "--tensor-parallel-size", "2")),
            ("profile name is empty or invalid", ("--profile", "../escape", *BASE[2:])),
            ("profile is not readable", ("--profile", "absent", *BASE[2:])),
            # The key must never reach a command line, and every launcher
            # option has one source, whatever its spelling.
            ("--extra-arg must not set --api-key", (*BASE, "--extra-arg=--api-key=secret")),
            ("--extra-arg must not set --host", (*BASE, "--extra-arg", "--host", "--extra-arg", "0.0.0.0")),
            ("--extra-arg must not set --max-model-len", (*BASE, "--extra-arg", "--max_model_len=1")),
        ]
        for message, args in cases:
            with self.subTest(message=message):
                self.assert_refused(message, *args)

    def test_a_profile_must_be_complete_and_consistent(self) -> None:
        cases = [
            ("VLLM_TASK is required in", dict(VLLM_TASK=None)),
            ("VLLM_NOTE is required in", dict(VLLM_NOTE=None)),
            ("VLLM_MODEL_PATH is not a directory", dict(VLLM_MODEL_PATH="/nonexistent")),
            ("VLLM_TASK must be generate or embed", dict(VLLM_TASK="classify")),
            ("VLLM_TENSOR_PARALLEL_SIZE must be a positive integer", dict(VLLM_TENSOR_PARALLEL_SIZE="two")),
            ("VLLM_ARGS is required in", dict(args=None)),
            ("VLLM_ARGS must be a Bash array", dict(args="--enforce-eager")),
            ("must not set --max-model-len", dict(args="(--max-model-len 8192)")),
        ]
        for message, overrides in cases:
            with self.subTest(message=message):
                self.repo.write_profile(**overrides)
                self.assert_refused(message, *BASE)

    def test_the_deployment_is_the_launcher_with_its_own_options(self) -> None:
        """One launch path: a deployment adds options, never a second argument list."""
        launch_args = (
            "--tensor-parallel-size", "2", "--max-model-len", "8192",
            "--extra-arg", "--max-num-seqs", "--extra-arg", "8",
        )
        self.repo.write_deployment(
            VLLM_GPUS="0,1", launch_args="(" + " ".join(launch_args) + ")"
        )
        deployment = self.launch()
        instance = self.launch(
            "--profile", "test-profile", "--gpus", "0,1", "--port", "65000", *launch_args
        )
        self.assertEqual(deployment, instance)


if __name__ == "__main__":
    unittest.main()
