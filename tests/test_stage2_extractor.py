from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Add workspace root to sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import torch

from generation.gen_ad_vlm import extract_dual_pass_representations


def test_stage2_end_to_end_smoke():
    print("=" * 60)
    print("RUNNING STAGE 2 VERIFICATION SUITE")
    print("=" * 60)

    config_path = r"D:\AD\Efficient_VLM_For_Autonomous_Driving_router\configs\repvit_t5_efficient_mini_kaggle_2gpu_router_vsa_k49.yaml"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "stage2_output"

        print(f"\n[Step 1] Running Dual-Pass Extraction (smoke=True, samples=4, batch_size=2)...")
        summary = extract_dual_pass_representations(
            config_path=config_path,
            checkpoint_path=None,
            output_dir=out_dir,
            max_samples=4,
            batch_size=2,
            device="cpu",
            smoke=True,
            max_new_tokens=16,
            output_attentions=True,
        )

        print("\n[Step 2] Verifying output artifacts on disk...")
        assert (out_dir / "hidden_states_vl.pt").exists(), "Missing hidden_states_vl.pt"
        assert (out_dir / "hidden_states_l.pt").exists(), "Missing hidden_states_l.pt"
        assert (out_dir / "visual_attention_weights.pt").exists(), "Missing visual_attention_weights.pt"
        assert (out_dir / "agreement_labels.pt").exists(), "Missing agreement_labels.pt"
        assert (out_dir / "metadata.json").exists(), "Missing metadata.json"
        assert (out_dir / "summary.json").exists(), "Missing summary.json"
        print("[PASS] All 6 output artifact files created successfully!")

        print("\n[Step 3] Verifying tensor dimensions and data integrity...")
        hs_vl = torch.load(out_dir / "hidden_states_vl.pt")
        hs_l = torch.load(out_dir / "hidden_states_l.pt")
        attn_vis = torch.load(out_dir / "visual_attention_weights.pt")
        agreements = torch.load(out_dir / "agreement_labels.pt")

        assert hs_vl.shape == (4, 5, 384), f"Unexpected hs_vl shape: {hs_vl.shape}"
        assert hs_l.shape == (4, 5, 384), f"Unexpected hs_l shape: {hs_l.shape}"
        assert attn_vis.shape == (4, 4), f"Unexpected attn_vis shape: {attn_vis.shape}"
        assert agreements.shape == (4,), f"Unexpected agreements shape: {agreements.shape}"
        assert agreements.dtype == torch.bool, f"Expected bool dtype, got {agreements.dtype}"

        # Kiểm tra tính hữu hạn không chứa NaN/Inf
        assert torch.isfinite(hs_vl).all(), "hs_vl contains NaN or Inf!"
        assert torch.isfinite(hs_l).all(), "hs_l contains NaN or Inf!"
        assert torch.isfinite(attn_vis).all(), "attn_vis contains NaN or Inf!"
        print("[PASS] Tensor dimensions [4, 5, 384], types, and numerical finiteness verified!")

        print("\n[Step 4] Verifying metadata and summary records...")
        meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
        assert len(meta) == 4, f"Expected 4 metadata records, got {len(meta)}"
        for item in meta:
            assert "sample_id" in item
            assert "question" in item
            assert "pred_vl" in item
            assert "pred_l" in item
            assert "is_same_prediction" in item

        assert summary["total_samples"] == 4
        assert summary["d_t_count"] + summary["d_vt_count"] == 4
        print(f"[PASS] Metadata and summary records verified!")
        print(f"  Summary: D_T count = {summary['d_t_count']}, D_VT count = {summary['d_vt_count']} ({summary['language_prior_ratio_pct']:.1f}%)")

    print("\n" + "=" * 60)
    print("STAGE 2 VERIFICATION PASSED 100% SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_stage2_end_to_end_smoke()
