"""Selection context diagnostics enrichment contract tests."""

import json
import tempfile
import unittest
from pathlib import Path


class SelectionContextDiagnosticsContractsTest(unittest.TestCase):
    def test_enriches_selector_json_with_sudo_memory_diagnostics_summary(self):
        from edge_vlm.selection_context_diagnostics import enrich_with_sudo_memory_diagnostics

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            selector_output = tmp_path / "qwen3-selector.json"
            sudo_output = tmp_path / "qwen3-selector.memory-diagnostics.sudo.json"
            selector_output.write_text(
                json.dumps(
                    {
                        "selection_id": "qwen3-vl-2b-instruct-auto",
                        "selected_variant_id": None,
                        "selected_reason": "no_usable_variant",
                        "memory_diagnostics_path": "qwen3-selector.memory-diagnostics.json",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sudo_output.write_text(
                json.dumps(
                    {
                        "summary": {
                            "debugfs_statuses": {"dma_buf_bufinfo": "readable"},
                            "debugfs_dma_buf_total_bytes": 0,
                            "debugfs_nvmap_clients_total_bytes": 0,
                        },
                        "debugfs_summary": {
                            "statuses": {"dma_buf_bufinfo": "readable"},
                            "dma_buf_bufinfo": {"object_count": 0, "total_bytes": 0},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            enriched = enrich_with_sudo_memory_diagnostics(
                selector_output=selector_output,
                sudo_memory_diagnostics_output=sudo_output,
            )
            written = json.loads(selector_output.read_text(encoding="utf-8"))

        self.assertEqual(enriched["sudo_memory_diagnostics_path"], str(sudo_output))
        self.assertEqual(
            enriched["sudo_memory_diagnostics_summary"]["debugfs_statuses"]["dma_buf_bufinfo"],
            "readable",
        )
        self.assertEqual(enriched["sudo_memory_diagnostics_summary"]["debugfs_dma_buf_total_bytes"], 0)
        self.assertEqual(written, enriched)


if __name__ == "__main__":
    unittest.main()
