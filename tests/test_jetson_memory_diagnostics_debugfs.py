"""Jetson memory diagnostics debugfs contract tests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class JetsonMemoryDiagnosticsDebugfsContractsTest(unittest.TestCase):
    def test_memory_diagnostics_handles_debugfs_permission_denied(self):
        from edge_vlm import jetson_memory_diagnostics

        with patch.object(jetson_memory_diagnostics.Path, "exists", side_effect=PermissionError("denied")):
            statuses = jetson_memory_diagnostics._debugfs_status(Path("/sys"))

        self.assertEqual(statuses["nvmap_iovmm_clients"]["status"], "unreadable")
        self.assertIn("denied", statuses["nvmap_iovmm_clients"]["error"])

    def test_memory_diagnostics_summarizes_readable_debugfs_totals(self):
        from edge_vlm.jetson_memory_diagnostics import capture_memory_diagnostics

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            proc_root = tmp_path / "proc"
            sys_root = tmp_path / "sys"
            (proc_root / "pressure").mkdir(parents=True)
            (sys_root / "kernel" / "debug" / "nvmap" / "iovmm").mkdir(parents=True)
            (sys_root / "kernel" / "debug" / "dma_buf").mkdir(parents=True)
            (sys_root / "kernel" / "debug" / "nvmap" / "iovmm" / "clients").write_text(
                "CLIENT                        PROCESS      PID        SIZE\n"
                "python                        llama       1234       1024K\n"
                "total                                                   1024K\n",
                encoding="utf-8",
            )
            (sys_root / "kernel" / "debug" / "nvmap" / "iovmm" / "allocations").write_text(
                "CLIENT                        PROCESS      PID        SIZE\n"
                "                              BASE        SIZE    FLAGS   REFS\n"
                "total                                                   2M\n",
                encoding="utf-8",
            )
            (sys_root / "kernel" / "debug" / "dma_buf" / "bufinfo").write_text(
                "\n"
                "Dma-buf Objects:\n"
                "size    flags    mode    count    exp_name    ino\n"
                "\n"
                "Total 3 objects, 4096 bytes\n",
                encoding="utf-8",
            )

            output = tmp_path / "memory-diagnostics.json"
            diagnostics = capture_memory_diagnostics(
                output,
                proc_root=proc_root,
                sys_root=sys_root,
                sample_tegrastats=False,
            )
            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(
            written["debugfs_summary"]["statuses"]["nvmap_iovmm_clients"],
            "readable",
        )
        self.assertEqual(written["debugfs_summary"]["statuses"]["cma_debug"], "missing")
        self.assertEqual(written["debugfs_summary"]["nvmap_iovmm_clients"]["total_bytes"], 1024 * 1024)
        self.assertEqual(written["debugfs_summary"]["nvmap_iovmm_allocations"]["total_bytes"], 2 * 1024 * 1024)
        self.assertEqual(written["debugfs_summary"]["dma_buf_bufinfo"]["object_count"], 3)
        self.assertEqual(written["debugfs_summary"]["dma_buf_bufinfo"]["total_bytes"], 4096)
        self.assertEqual(diagnostics["summary"]["debugfs_statuses"]["dma_buf_bufinfo"], "readable")
        self.assertEqual(diagnostics["summary"]["debugfs_dma_buf_object_count"], 3)
        self.assertEqual(diagnostics["summary"]["debugfs_dma_buf_total_bytes"], 4096)
        self.assertEqual(diagnostics["summary"]["debugfs_nvmap_clients_total_bytes"], 1024 * 1024)
        self.assertEqual(diagnostics["summary"]["debugfs_nvmap_allocations_total_bytes"], 2 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
