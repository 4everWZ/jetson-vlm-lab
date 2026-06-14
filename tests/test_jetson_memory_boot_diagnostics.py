"""Boot-time memory configuration diagnostics contract tests."""

import json
import tempfile
import unittest
from pathlib import Path


class JetsonMemoryBootDiagnosticsTest(unittest.TestCase):
    def test_memory_diagnostics_captures_cmdline_cma_and_reserved_memory_nodes(self):
        from edge_vlm.jetson_memory_diagnostics import capture_memory_diagnostics

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            proc_root = tmp_path / "proc"
            sys_root = tmp_path / "sys"
            reserved_root = sys_root / "firmware" / "devicetree" / "base" / "reserved-memory"
            cma_node = reserved_root / "linux,cma"
            ramoops_node = reserved_root / "ramoops@90000000"
            cma_node.mkdir(parents=True)
            ramoops_node.mkdir()
            proc_root.mkdir()
            (proc_root / "cmdline").write_text(
                "root=/dev/mmcblk0p1 rw quiet cma=256M@0-4G coherent_pool=4M\n",
                encoding="utf-8",
            )
            (cma_node / "compatible").write_bytes(b"shared-dma-pool\x00")
            (cma_node / "status").write_bytes(b"okay\x00")
            (cma_node / "reusable").write_bytes(b"")
            (cma_node / "reg").write_bytes(bytes.fromhex("00000000900000000000000010000000"))
            (ramoops_node / "compatible").write_bytes(b"ramoops\x00")
            (ramoops_node / "size").write_bytes(bytes.fromhex("0000000000100000"))

            output = tmp_path / "memory-diagnostics.json"
            diagnostics = capture_memory_diagnostics(
                output,
                proc_root=proc_root,
                sys_root=sys_root,
                sample_tegrastats=False,
            )
            written = json.loads(output.read_text(encoding="utf-8"))

        boot_memory = written["boot_memory"]
        self.assertEqual(boot_memory["cmdline"]["cma_token"], "cma=256M@0-4G")
        self.assertEqual(
            boot_memory["cmdline"]["raw"],
            "root=/dev/mmcblk0p1 rw quiet cma=256M@0-4G coherent_pool=4M",
        )
        self.assertTrue(boot_memory["reserved_memory"]["available"])
        self.assertEqual(boot_memory["reserved_memory"]["node_count"], 2)
        self.assertEqual(
            [node["name"] for node in boot_memory["reserved_memory"]["nodes"]],
            ["linux,cma", "ramoops@90000000"],
        )
        self.assertEqual(
            boot_memory["reserved_memory"]["nodes"][0]["properties"]["compatible"],
            ["shared-dma-pool"],
        )
        self.assertEqual(boot_memory["reserved_memory"]["nodes"][0]["properties"]["status"], ["okay"])
        self.assertTrue(boot_memory["reserved_memory"]["nodes"][0]["properties"]["reusable"])
        self.assertEqual(
            boot_memory["reserved_memory"]["nodes"][0]["properties"]["reg_hex"],
            "00000000900000000000000010000000",
        )
        self.assertEqual(
            boot_memory["reserved_memory"]["nodes"][1]["properties"]["size_hex"],
            "0000000000100000",
        )
        self.assertEqual(diagnostics["summary"]["boot_cmdline_cma_token"], "cma=256M@0-4G")
        self.assertEqual(diagnostics["summary"]["reserved_memory_node_count"], 2)
        self.assertEqual(
            diagnostics["summary"]["reserved_memory_names"],
            ["linux,cma", "ramoops@90000000"],
        )

    def test_memory_diagnostics_marks_missing_boot_memory_sources_unavailable(self):
        from edge_vlm.jetson_memory_diagnostics import capture_memory_diagnostics

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            proc_root = tmp_path / "proc"
            sys_root = tmp_path / "sys"
            proc_root.mkdir()
            sys_root.mkdir()

            output = tmp_path / "memory-diagnostics.json"
            diagnostics = capture_memory_diagnostics(
                output,
                proc_root=proc_root,
                sys_root=sys_root,
                sample_tegrastats=False,
            )

        self.assertFalse(diagnostics["boot_memory"]["cmdline"]["available"])
        self.assertIsNone(diagnostics["boot_memory"]["cmdline"]["cma_token"])
        self.assertFalse(diagnostics["boot_memory"]["reserved_memory"]["available"])
        self.assertEqual(diagnostics["boot_memory"]["reserved_memory"]["nodes"], [])
        self.assertIsNone(diagnostics["summary"]["boot_cmdline_cma_token"])
        self.assertEqual(diagnostics["summary"]["reserved_memory_node_count"], 0)


if __name__ == "__main__":
    unittest.main()
