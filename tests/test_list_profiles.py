"""The profile listing: what it reports, and that it never reports a broken profile."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from _repo import Repo


class ListProfilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Repo(Path(self._tmp.name))

    def run_listing(self) -> subprocess.CompletedProcess:
        return self.repo.run("list-profiles")

    def test_every_profile_is_listed_with_its_facts(self) -> None:
        (self.repo.model / "extra.safetensors").write_bytes(b"\0" * 23)
        (self.repo.model / "config.json").write_text("{}", encoding="utf-8")
        self.repo.write_profile(
            name="other", VLLM_TASK="embed", VLLM_KNOWLEDGE_CUTOFF="2025-12",
            VLLM_NOTE='quotes " and \\ survive',
        )
        result = self.run_listing()
        self.assertEqual(result.returncode, 0, result.stderr)
        listing = json.loads(result.stdout)
        self.assertEqual(
            listing,
            [
                {
                    "name": "other",
                    "served_model_name": "test-model",
                    "task": "embed",
                    "model_path": str(self.repo.model),
                    "weights_bytes": 123,
                    "tensor_parallel_size": 1,
                    "max_model_len": 4096,
                    "release_date": "2026-01",
                    "knowledge_cutoff": "2025-12",
                    "note": 'quotes " and \\ survive',
                },
                {
                    "name": "test-profile",
                    "served_model_name": "test-model",
                    "task": "generate",
                    "model_path": str(self.repo.model),
                    "weights_bytes": 123,
                    "tensor_parallel_size": 1,
                    "max_model_len": 4096,
                    "release_date": "2026-01",
                    "knowledge_cutoff": None,
                    "note": "a test model",
                },
            ],
        )

    def test_a_broken_profile_fails_the_whole_listing(self) -> None:
        cases = {
            "VLLM_RELEASE_DATE is required in": dict(VLLM_RELEASE_DATE=None),
            "no safetensors weights in": dict(VLLM_MODEL_PATH=str(self.repo.root / "scripts")),
        }
        for message, overrides in cases.items():
            with self.subTest(message=message):
                self.repo.write_profile(name="broken", **overrides)
                result = self.run_listing()
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(message, result.stderr)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
