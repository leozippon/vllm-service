"""Ownership and liveness rules of the /dev/shm offload-region sweep.

Deleting a region that a live engine still maps would corrupt that engine, so
these tests drive the real script against real files and real processes.

The ownership branch (a region owned by another tenant is left alone) cannot be
exercised from a single UID: the check is the shell's own ``[[ -O ]]`` and the
test suite cannot create a file it does not own. It is additionally enforced by
the kernel, because /dev/shm is sticky and vLLM creates regions mode 0600.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SWEEP = Path(__file__).resolve().parent.parent / "scripts" / "reclaim-offload-regions"

HOLD_DESCRIPTOR = """
import os, sys, time
fd = os.open(sys.argv[1], os.O_RDWR)
sys.stdout.write("ready\\n"); sys.stdout.flush()
time.sleep(60)
"""

# Map the region and close the descriptor, so only /proc/<pid>/maps shows it.
HOLD_MAPPING = """
import ctypes, os, sys, time
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
fd = os.open(sys.argv[1], os.O_RDWR)
addr = libc.mmap(None, 4096, 3, 1, fd, 0)
os.close(fd)
assert addr not in (None, ctypes.c_void_p(-1).value)
sys.stdout.write("ready\\n"); sys.stdout.flush()
time.sleep(60)
"""


def _sweep(directory: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["VLLM_OFFLOAD_DIR"] = str(directory)
    return subprocess.run(
        [SWEEP], env=env, check=False, capture_output=True, text=True
    )


class _Holder:
    """A real process holding a region, so liveness is not simulated."""

    def __init__(self, program: str, region: Path) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-c", program, str(region)],
            stdout=subprocess.PIPE,
            text=True,
        )

    def __enter__(self) -> "_Holder":
        assert self.proc.stdout is not None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.proc.stdout.readline().strip() == "ready":
                return self
            if self.proc.poll() is not None:
                raise AssertionError("region holder exited before signalling ready")
        raise AssertionError("region holder never signalled ready")

    def __exit__(self, *exc) -> None:
        self.proc.kill()
        self.proc.wait()
        if self.proc.stdout is not None:
            self.proc.stdout.close()


class ReclaimOffloadRegionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _region(self, uuid: str, size: int = 4096) -> Path:
        region = self.dir / f"vllm_offload_{uuid}.mmap"
        region.write_bytes(b"\0" * size)
        return region

    def test_orphan_is_reclaimed_and_other_files_are_left_alone(self) -> None:
        orphan = self._region("11111111-1111-1111-1111-111111111111")
        bystanders = [
            self.dir / "unrelated.mmap",
            self.dir / "psm_deadbeef",
            self.dir / "vllm_offload_22222222.mmap.bak",
        ]
        for path in bystanders:
            path.write_bytes(b"x")

        result = _sweep(self.dir)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(orphan.exists(), result.stdout)
        self.assertIn("reclaimed", result.stdout)
        for path in bystanders:
            self.assertTrue(path.exists(), f"{path} was removed")

    def test_region_held_by_an_open_descriptor_survives(self) -> None:
        region = self._region("33333333-3333-3333-3333-333333333333")
        with _Holder(HOLD_DESCRIPTOR, region):
            result = _sweep(self.dir)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(region.exists(), result.stdout)
        self.assertIn("held open or mapped", result.stdout)

    def test_mapped_region_without_an_open_descriptor_survives(self) -> None:
        region = self._region("44444444-4444-4444-4444-444444444444")
        with _Holder(HOLD_MAPPING, region):
            self.assertEqual(
                self._descriptor_holders(region), 0, "expected a mapping-only holder"
            )
            result = _sweep(self.dir)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(region.exists(), result.stdout)

    def test_symlinked_region_is_never_followed(self) -> None:
        target = self.dir / "target-that-must-survive"
        target.write_bytes(b"x")
        link = self.dir / "vllm_offload_55555555-5555-5555-5555-555555555555.mmap"
        link.symlink_to(target)

        result = _sweep(self.dir)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(target.exists(), result.stdout)
        self.assertTrue(link.is_symlink(), result.stdout)
        self.assertIn("not a regular file", result.stdout)

    @staticmethod
    def _descriptor_holders(region: Path) -> int:
        count = 0
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                for fd in (entry / "fd").iterdir():
                    if os.readlink(fd) == str(region):
                        count += 1
            except OSError:
                continue
        return count


if __name__ == "__main__":
    unittest.main()
