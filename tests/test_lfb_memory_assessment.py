"""LFB memory assessment contract tests."""

import unittest


class LfbMemoryAssessmentContractsTest(unittest.TestCase):
    def test_assessment_summarizes_lfb_deficit_and_allocator_evidence(self):
        from edge_vlm.lfb_memory_assessment import assess_selection_memory

        selection = {
            "selected_variant_id": None,
            "selected_reason": "no_usable_variant",
            "preflight": {"tegrastats": {"lfb": {"free_blocks": 69, "block_mb": 4}}},
            "candidates": [
                {
                    "variant_id": "qwen3-vl-2b-instruct-q4-smoke",
                    "min_lfb_blocks": 150,
                    "block_reasons": ["lfb_free_blocks 69 < required 150"],
                    "usable": False,
                },
                {
                    "variant_id": "qwen3-vl-2b-instruct-q8-smoke",
                    "min_lfb_blocks": 100,
                    "block_reasons": ["lfb_free_blocks 69 < required 100"],
                    "usable": False,
                },
            ],
        }
        diagnostics_summary = {
            "cma_total_kb": 262144,
            "cma_free_kb": 221296,
            "debugfs_statuses": {
                "nvmap_iovmm_clients": "readable",
                "nvmap_iovmm_allocations": "readable",
                "dma_buf_bufinfo": "readable",
            },
            "debugfs_dma_buf_total_bytes": 0,
            "debugfs_nvmap_clients_total_bytes": 0,
            "debugfs_nvmap_allocations_total_bytes": 0,
            "tegrastats_lfb_free_blocks": 69,
            "tegrastats_lfb_block_mb": 4,
            "boot_cmdline_cma_token": None,
            "reserved_memory_node_count": 7,
            "reserved_memory_names": [
                "camdbg_carveout",
                "linux,cma",
                "vpr-carveout",
            ],
        }

        assessment = assess_selection_memory(
            selection=selection,
            diagnostics_summary=diagnostics_summary,
            diagnostics_source="sudo",
        )

        self.assertEqual(assessment["status"], "lfb_gate_not_met")
        self.assertEqual(assessment["diagnostics_source"], "sudo")
        self.assertEqual(assessment["observed_lfb_free_blocks"], 69)
        self.assertEqual(assessment["observed_lfb_block_mb"], 4)
        self.assertEqual(assessment["required_lfb_blocks_max"], 150)
        self.assertEqual(assessment["lfb_free_block_deficit_max"], 81)
        self.assertEqual(assessment["required_lfb_bytes_max"], 150 * 4 * 1024 * 1024)
        self.assertEqual(assessment["observed_lfb_bytes"], 69 * 4 * 1024 * 1024)
        self.assertEqual(assessment["cma_free_kb"], 221296)
        self.assertEqual(assessment["debugfs_total_tracked_bytes"], 0)
        self.assertIsNone(assessment["boot_cmdline_cma_token"])
        self.assertEqual(assessment["reserved_memory_node_count"], 7)
        self.assertEqual(
            assessment["reserved_memory_names"],
            ["camdbg_carveout", "linux,cma", "vpr-carveout"],
        )
        self.assertTrue(assessment["linux_cma_reserved_memory_present"])
        self.assertEqual(
            assessment["candidate_lfb_deficits"],
            {
                "qwen3-vl-2b-instruct-q4-smoke": 81,
                "qwen3-vl-2b-instruct-q8-smoke": 31,
            },
        )
        self.assertIn("lfb_below_required", assessment["signals"])
        self.assertIn("debugfs_tracked_allocations_zero", assessment["signals"])
        self.assertIn("cma_free_below_required_lfb_bytes", assessment["signals"])
        self.assertEqual(assessment["selection_implication"], "no_candidate_meets_lfb_gate")


if __name__ == "__main__":
    unittest.main()
