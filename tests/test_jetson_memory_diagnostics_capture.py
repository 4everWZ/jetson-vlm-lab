"""Jetson memory diagnostics capture artifact contract tests."""

import json
import tempfile
import unittest
from pathlib import Path


class JetsonMemoryDiagnosticsCaptureContractsTest(unittest.TestCase):
    def test_memory_diagnostics_captures_fragmentation_and_pressure_context(self):
        from edge_vlm.jetson_memory_diagnostics import capture_memory_diagnostics

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            proc_root = tmp_path / "proc"
            sys_root = tmp_path / "sys"
            (proc_root / "pressure").mkdir(parents=True)
            (proc_root / "101").mkdir()
            (proc_root / "202").mkdir()
            (sys_root / "block" / "zram0").mkdir(parents=True)
            (sys_root / "kernel" / "debug" / "dma_buf").mkdir(parents=True)
            (sys_root / "kernel" / "debug" / "dma_buf" / "bufinfo").write_text("dmabuf summary\n", encoding="utf-8")
            (proc_root / "meminfo").write_text(
                "\n".join(
                    [
                        "MemTotal:        7802712 kB",
                        "MemFree:         6593296 kB",
                        "MemAvailable:    6940680 kB",
                        "SwapTotal:      12289948 kB",
                        "SwapFree:       12289948 kB",
                        "CmaTotal:         262144 kB",
                        "CmaFree:          221296 kB",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (proc_root / "buddyinfo").write_text(
                "Node 0, zone   Normal   1552   1175   1138    550    246     97     41     16     39     19     67     38    236\n",
                encoding="utf-8",
            )
            (proc_root / "vmstat").write_text(
                "\n".join(["compact_success 7", "compact_fail 3", "pgscan_kswapd 11"]) + "\n",
                encoding="utf-8",
            )
            (proc_root / "pressure" / "memory").write_text(
                "some avg10=0.00 avg60=0.00 avg300=0.00 total=123\n"
                "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
                encoding="utf-8",
            )
            (proc_root / "swaps").write_text(
                "Filename\tType\tSize\tUsed\tPriority\n"
                "/swfile\tfile\t8388604\t0\t-2\n"
                "/dev/zram0\tpartition\t650224\t12\t5\n",
                encoding="utf-8",
            )
            (proc_root / "101" / "status").write_text(
                "Name:\tpython3\nState:\tS (sleeping)\nUid:\t1000\t1000\t1000\t1000\nVmRSS:\t204800 kB\n",
                encoding="utf-8",
            )
            (proc_root / "101" / "cmdline").write_bytes(b"python3\x00server.py\x00")
            (proc_root / "202" / "status").write_text(
                "Name:\tdockerd\nState:\tS (sleeping)\nUid:\t0\t0\t0\t0\nVmRSS:\t102400 kB\n",
                encoding="utf-8",
            )
            (proc_root / "202" / "cmdline").write_bytes(b"dockerd\x00")
            (sys_root / "block" / "zram0" / "disksize").write_text("665833472\n", encoding="utf-8")
            (sys_root / "block" / "zram0" / "mm_stat").write_text("4096 2048 8192 0 0 0 0 0\n", encoding="utf-8")

            output = tmp_path / "memory-diagnostics.json"
            diagnostics = capture_memory_diagnostics(
                output,
                proc_root=proc_root,
                sys_root=sys_root,
                top_process_limit=2,
                sample_tegrastats=False,
            )
            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(diagnostics["schema_version"], 1)
        self.assertEqual(written["meminfo_kb"]["CmaFree"], 221296)
        self.assertEqual(written["buddyinfo"]["max_order_with_free_block"], 12)
        self.assertEqual(written["swap"]["total_kb"], 9038828)
        self.assertEqual(written["swap"]["used_kb"], 12)
        self.assertEqual(written["zram"]["devices"][0]["name"], "zram0")
        self.assertEqual(written["zram"]["devices"][0]["disksize_bytes"], 665833472)
        self.assertEqual(written["vmstat"]["compact_success"], 7)
        self.assertEqual(written["pressure"]["memory"][0]["kind"], "some")
        self.assertEqual(written["top_rss_processes"][0]["pid"], 101)
        self.assertEqual(written["top_rss_processes"][0]["rss_kb"], 204800)
        self.assertEqual(written["debugfs"]["dma_buf_bufinfo"]["status"], "readable")
        self.assertEqual(written["summary"]["cma_free_kb"], 221296)
        self.assertEqual(written["summary"]["swap_used_kb"], 12)
        self.assertEqual(written["summary"]["top_rss_processes"][0]["command"], "python3 server.py")


if __name__ == "__main__":
    unittest.main()
