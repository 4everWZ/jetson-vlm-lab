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
            boot_root = tmp_path
            reserved_root = sys_root / "firmware" / "devicetree" / "base" / "reserved-memory"
            cma_node = reserved_root / "linux,cma"
            ramoops_node = reserved_root / "ramoops@90000000"
            cma_node.mkdir(parents=True)
            ramoops_node.mkdir()
            proc_root.mkdir()
            extlinux = boot_root / "boot" / "extlinux" / "extlinux.conf"
            extlinux.parent.mkdir(parents=True)
            extlinux.write_text(
                "\n".join(
                    [
                        "TIMEOUT 30",
                        "DEFAULT primary",
                        "LABEL primary",
                        "      MENU LABEL primary kernel",
                        "      LINUX /boot/Image",
                        "      APPEND root=/dev/mmcblk0p1 rw cma=768M quiet",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (reserved_root / "#address-cells").write_bytes(bytes.fromhex("00000002"))
            (reserved_root / "#size-cells").write_bytes(bytes.fromhex("00000002"))
            (proc_root / "cmdline").write_text(
                "root=/dev/mmcblk0p1 rw quiet cma=256M@0-4G coherent_pool=4M\n",
                encoding="utf-8",
            )
            (cma_node / "compatible").write_bytes(b"shared-dma-pool\x00")
            (cma_node / "status").write_bytes(b"okay\x00")
            (cma_node / "reusable").write_bytes(b"")
            (cma_node / "reg").write_bytes(bytes.fromhex("00000000900000000000000010000000"))
            (cma_node / "size").write_bytes(bytes.fromhex("0000000010000000"))
            (ramoops_node / "compatible").write_bytes(b"ramoops\x00")
            (ramoops_node / "size").write_bytes(bytes.fromhex("0000000000100000"))

            output = tmp_path / "memory-diagnostics.json"
            diagnostics = capture_memory_diagnostics(
                output,
                proc_root=proc_root,
                sys_root=sys_root,
                boot_root=boot_root,
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
        self.assertEqual(boot_memory["reserved_memory"]["address_cells"], 2)
        self.assertEqual(boot_memory["reserved_memory"]["size_cells"], 2)
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
        self.assertEqual(boot_memory["reserved_memory"]["nodes"][0]["properties"]["size_bytes"], 256 * 1024 * 1024)
        self.assertEqual(
            boot_memory["reserved_memory"]["nodes"][0]["properties"]["reg_regions"],
            [{"address_bytes": 0x90000000, "size_bytes": 256 * 1024 * 1024}],
        )
        self.assertEqual(
            boot_memory["reserved_memory"]["nodes"][1]["properties"]["size_hex"],
            "0000000000100000",
        )
        self.assertEqual(diagnostics["summary"]["boot_cmdline_cma_token"], "cma=256M@0-4G")
        self.assertEqual(boot_memory["boot_config"]["paths"][0]["status"], "readable")
        self.assertEqual(boot_memory["boot_config"]["paths"][0]["append_line_count"], 1)
        self.assertEqual(boot_memory["boot_config"]["paths"][0]["default_labels"], ["primary"])
        self.assertEqual(boot_memory["boot_config"]["paths"][0]["labels"], ["primary"])
        self.assertEqual(boot_memory["boot_config"]["paths"][0]["cma_tokens"], ["cma=768M"])
        self.assertEqual(diagnostics["summary"]["boot_config_readable_paths"], [str(extlinux)])
        self.assertEqual(diagnostics["summary"]["boot_config_cma_tokens"], ["cma=768M"])
        self.assertTrue(diagnostics["summary"]["boot_config_has_cma_token"])
        self.assertEqual(diagnostics["summary"]["boot_config_append_line_count"], 1)
        self.assertEqual(diagnostics["summary"]["reserved_memory_node_count"], 2)
        self.assertEqual(
            diagnostics["summary"]["reserved_memory_names"],
            ["linux,cma", "ramoops@90000000"],
        )
        self.assertEqual(diagnostics["summary"]["linux_cma_reserved_size_bytes"], 256 * 1024 * 1024)

    def test_memory_diagnostics_marks_missing_boot_memory_sources_unavailable(self):
        from edge_vlm.jetson_memory_diagnostics import capture_memory_diagnostics

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            proc_root = tmp_path / "proc"
            sys_root = tmp_path / "sys"
            boot_root = tmp_path
            proc_root.mkdir()
            sys_root.mkdir()

            output = tmp_path / "memory-diagnostics.json"
            diagnostics = capture_memory_diagnostics(
                output,
                proc_root=proc_root,
                sys_root=sys_root,
                boot_root=boot_root,
                sample_tegrastats=False,
            )

        self.assertFalse(diagnostics["boot_memory"]["cmdline"]["available"])
        self.assertIsNone(diagnostics["boot_memory"]["cmdline"]["cma_token"])
        self.assertFalse(diagnostics["boot_memory"]["reserved_memory"]["available"])
        self.assertEqual(diagnostics["boot_memory"]["reserved_memory"]["nodes"], [])
        self.assertFalse(diagnostics["boot_memory"]["boot_config"]["available"])
        self.assertEqual(diagnostics["boot_memory"]["boot_config"]["paths"][0]["status"], "missing")
        self.assertIsNone(diagnostics["summary"]["boot_cmdline_cma_token"])
        self.assertEqual(diagnostics["summary"]["boot_config_readable_paths"], [])
        self.assertEqual(diagnostics["summary"]["boot_config_cma_tokens"], [])
        self.assertFalse(diagnostics["summary"]["boot_config_has_cma_token"])
        self.assertEqual(diagnostics["summary"]["boot_config_append_line_count"], 0)
        self.assertEqual(diagnostics["summary"]["reserved_memory_node_count"], 0)


if __name__ == "__main__":
    unittest.main()
