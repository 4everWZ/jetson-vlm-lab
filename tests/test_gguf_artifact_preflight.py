"""Machine-readable GGUF artifact preflight contracts."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from edge_vlm.gguf_artifacts import inspect_gguf_artifacts


def _run_check_plan(plan: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "edge_vlm.gguf_artifacts",
            "check-plan",
            "--plan",
            str(plan),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        encoding="utf-8",
    )


class GgufArtifactPreflightContractsTest(unittest.TestCase):

    def test_inspect_gguf_artifacts_reports_ok_status_and_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model = tmp_path / "model.gguf"
            mmproj = tmp_path / "mmproj.gguf"
            model.write_bytes(b"GGUFmodel")
            mmproj.write_bytes(b"GGUFmmproj")

            manifest = inspect_gguf_artifacts(
                [
                    ("model", model),
                    ("mmproj", mmproj),
                ]
            )

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["status"], "ok")
        self.assertEqual([artifact["role"] for artifact in manifest["artifacts"]], ["model", "mmproj"])
        self.assertEqual(manifest["artifacts"][0]["status"], "ok")
        self.assertTrue(manifest["artifacts"][0]["gguf_magic"])
        self.assertEqual(manifest["artifacts"][0]["size_bytes"], len(b"GGUFmodel"))
        self.assertEqual(manifest["artifacts"][1]["status"], "ok")

    def test_cli_writes_failed_manifest_for_missing_and_invalid_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bad_model = tmp_path / "bad-model.gguf"
            missing_mmproj = tmp_path / "missing-mmproj.gguf"
            output = tmp_path / "gguf-artifacts.json"
            bad_model.write_bytes(b"not-a-gguf")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "edge_vlm.gguf_artifacts",
                    "check",
                    "--artifact",
                    "model",
                    str(bad_model),
                    "--artifact",
                    "mmproj",
                    str(missing_mmproj),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
            )

            manifest = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["artifacts"][0]["status"], "invalid_magic")
        self.assertFalse(manifest["artifacts"][0]["gguf_magic"])
        self.assertEqual(manifest["artifacts"][1]["status"], "missing")
        self.assertFalse(manifest["artifacts"][1]["exists"])
        self.assertIn("gguf_artifact_check=failed", result.stderr)

    def test_cli_prints_manifest_to_stdout_when_output_is_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUFmodel")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "edge_vlm.gguf_artifacts",
                    "check",
                    "--artifact",
                    "model",
                    str(model),
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")

    def test_jetson_wrapper_sets_pythonpath_for_preflight_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            output = Path(tmp) / "manifest.json"
            model.write_bytes(b"GGUFmodel")

            result = subprocess.run(
                [
                    "scripts/jetson/check_gguf_artifacts.sh",
                    "check",
                    "--artifact",
                    "model",
                    str(model),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                encoding="utf-8",
            )

            manifest = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(manifest["status"], "ok")
        self.assertIn("gguf_artifact_check=ok", result.stderr)

    def test_check_plan_writes_grouped_variant_artifact_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model = tmp_path / "model.gguf"
            missing_mmproj = tmp_path / "missing-mmproj.gguf"
            plan = tmp_path / "plan.json"
            output = tmp_path / "plan-gguf-artifacts.json"
            model.write_bytes(b"GGUFmodel")
            plan.write_text(
                json.dumps(
                    {
                        "run_prefix": "unit",
                        "variants": [
                            {
                                "run_id": "unit-variant",
                                "variant": {"id": "variant"},
                                "artifact_preflight": {
                                    "artifacts": [
                                        {"role": "model", "path": str(model)},
                                        {"role": "mmproj", "path": str(missing_mmproj)},
                                    ]
                                },
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = _run_check_plan(plan, output)

            manifest = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["variant_count"], 1)
        self.assertEqual(manifest["failed_variant_count"], 1)
        self.assertEqual(manifest["variants"][0]["run_id"], "unit-variant")
        self.assertEqual(manifest["variants"][0]["artifact_manifest"]["status"], "failed")
        self.assertEqual(manifest["variants"][0]["artifact_manifest"]["artifacts"][1]["status"], "missing")
        self.assertIn("gguf_artifact_plan_check=failed", result.stderr)

    def test_check_plan_writes_ok_grouped_variant_artifact_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model = tmp_path / "model.gguf"
            mmproj = tmp_path / "mmproj.gguf"
            plan = tmp_path / "plan.json"
            output = tmp_path / "plan-gguf-artifacts.json"
            model.write_bytes(b"GGUFmodel")
            mmproj.write_bytes(b"GGUFmmproj")
            plan.write_text(
                json.dumps(
                    {
                        "run_prefix": "unit",
                        "variants": [
                            {
                                "run_id": "unit-variant",
                                "variant": {"id": "variant"},
                                "artifact_preflight": {
                                    "artifacts": [
                                        {"role": "model", "path": str(model)},
                                        {"role": "mmproj", "path": str(mmproj)},
                                    ]
                                },
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = _run_check_plan(plan, output)
            manifest = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(manifest["status"], "ok")
        self.assertEqual(manifest["variant_count"], 1)
        self.assertEqual(manifest["failed_variant_count"], 0)
        self.assertEqual(manifest["variants"][0]["artifact_manifest"]["status"], "ok")
        self.assertEqual(manifest["variants"][0]["artifact_manifest"]["artifact_count"], 2)
        self.assertIn("gguf_artifact_plan_check=ok", result.stderr)

    def test_check_plan_marks_variant_without_declared_artifacts_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan = tmp_path / "plan.json"
            output = tmp_path / "plan-gguf-artifacts.json"
            plan.write_text(
                json.dumps(
                    {
                        "run_prefix": "unit",
                        "variants": [
                            {
                                "run_id": "unit-variant",
                                "variant": {"id": "variant"},
                                "artifact_preflight": {"artifacts": []},
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = _run_check_plan(plan, output)
            manifest = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failed_variant_count"], 1)
        artifact_manifest = manifest["variants"][0]["artifact_manifest"]
        self.assertEqual(artifact_manifest["status"], "unavailable")
        self.assertEqual(artifact_manifest["reason"], "no_artifacts_declared")
        self.assertEqual(artifact_manifest["artifact_count"], 0)
        self.assertIn("gguf_artifact_plan_check=failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
