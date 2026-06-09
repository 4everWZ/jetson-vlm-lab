"""Formal Jetson benchmark wrapper shell contract tests."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class FormalBenchmarkWrapperTest(unittest.TestCase):
    def test_formal_jetson_benchmark_wrapper_dry_run_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cases = tmp_path / "cases.jsonl"
            output = tmp_path / "bench.jsonl"
            summary = tmp_path / "bench.md"
            metadata = tmp_path / "bench.manifest.json"
            cases.write_text(
                json.dumps({"id": "text_case", "input_type": "text", "prompt": "Say hi."}) + "\n",
                encoding="utf-8",
            )
            env = {
                **os.environ,
                "PYTHONPATH": "src",
                "EDGE_VLM_FORMAL_RUN_ID": "formal-wrapper-unit",
                "EDGE_VLM_CONFIG": "configs/models/minicpmv46_q4.yaml",
                "EDGE_VLM_CASES": str(cases),
                "EDGE_VLM_OUTPUT": str(output),
                "EDGE_VLM_SUMMARY_OUTPUT": str(summary),
                "EDGE_VLM_METADATA_OUTPUT": str(metadata),
                "EDGE_VLM_TRIAL_COUNT": "2",
                "EDGE_VLM_MAX_TOKENS": "8",
                "EDGE_VLM_TEMPERATURE": "0",
                "EDGE_VLM_FORMAL_DRY_RUN": "1",
                "EDGE_VLM_SKIP_TEGRASTATS": "1",
            }
            result = subprocess.run(
                ["bash", "scripts/jetson/run_formal_benchmark.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            manifest = json.loads(metadata.read_text(encoding="utf-8"))

        self.assertEqual(len(records), 2)
        self.assertEqual(manifest["run_id"], "formal-wrapper-unit")
        self.assertEqual(manifest["cases_written"], 2)
        self.assertEqual(manifest["jetson"]["tegrastats_log"], None)
        self.assertEqual(manifest["jetson"]["tegrastats_status"], "skipped")

    def test_formal_jetson_benchmark_wrapper_prefixes_tegrastats_with_utc_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            bin_dir.mkdir()
            fake_tegrastats = bin_dir / "tegrastats"
            fake_tegrastats.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "printf '%s\\n' 'RAM 1000/7620MB (lfb 200x4MB) CPU [10%@1000] GR3D_FREQ 20%@[1020] EMC_FREQ 30%@3199 gpu@40.0C VDD_IN 8000mW/7000mW'",
                        "sleep 5",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(fake_tegrastats, 0o755)
            cases = tmp_path / "cases.jsonl"
            output = tmp_path / "bench.jsonl"
            summary = tmp_path / "bench.md"
            metadata = tmp_path / "bench.manifest.json"
            tegrastats_log = tmp_path / "tegrastats.log"
            cases.write_text(
                json.dumps({"id": "text_case", "input_type": "text", "prompt": "Say hi."}) + "\n",
                encoding="utf-8",
            )
            env = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "PYTHONPATH": "src",
                "EDGE_VLM_FORMAL_RUN_ID": "formal-wrapper-timestamp-unit",
                "EDGE_VLM_CONFIG": "configs/models/minicpmv46_q4.yaml",
                "EDGE_VLM_CASES": str(cases),
                "EDGE_VLM_OUTPUT": str(output),
                "EDGE_VLM_SUMMARY_OUTPUT": str(summary),
                "EDGE_VLM_METADATA_OUTPUT": str(metadata),
                "EDGE_VLM_TEGRASTATS_LOG": str(tegrastats_log),
                "EDGE_VLM_TRIAL_COUNT": "1",
                "EDGE_VLM_MAX_TOKENS": "8",
                "EDGE_VLM_TEMPERATURE": "0",
                "EDGE_VLM_FORMAL_DRY_RUN": "1",
            }
            result = subprocess.run(
                ["bash", "scripts/jetson/run_formal_benchmark.sh"],
                check=False,
                capture_output=True,
                encoding="utf-8",
                env=env,
            )
            log_text = tegrastats_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(log_text, r"^\d{4}-\d{2}-\d{2}T.*Z RAM 1000/7620MB")


if __name__ == "__main__":
    unittest.main()
