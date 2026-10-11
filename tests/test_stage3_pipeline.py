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
from plot_divergences_ad import run_divergence_analysis


def test_stage3_end_to_end_pipeline():
    print("=" * 60)
    print("RUNNING STAGE 3 INTEGRATION PIPELINE TEST")
    print("=" * 60)

    config_path = r"D:\AD\Efficient_VLM_For_Autonomous_Driving_router\configs\repvit_t5_efficient_mini_kaggle_2gpu_router_vsa_k49.yaml"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        stage2_out = tmp_path / "stage2_artifacts"
        stage3_out = tmp_path / "stage3_plots"

        # 1. Run Stage 2 smoke extractor to produce real test tensors
        print("\n[Step 1] Generating Stage 2 input tensors (smoke=True, samples=4)...")
        extract_dual_pass_representations(
            config_path=config_path,
            output_dir=stage2_out,
            max_samples=4,
            batch_size=2,
            device="cpu",
            smoke=True,
            max_new_tokens=16,
            output_attentions=True,
        )

        # 2. Run Stage 3 Analysis & Visualization Engine
        print("\n[Step 2] Running Stage 3 Divergence Analysis & Visualization Engine...")
        results = run_divergence_analysis(
            input_dir=stage2_out,
            output_dir=stage3_out,
            model_name="Router VSA (K=49) + RepViT + T5-Mini",
            vip_layer="auto",
            metric="cosine",
            dpi=150,
        )

        # 3. Verify all artifacts exist on disk
        print("\n[Step 3] Verifying generated analysis artifacts on disk...")
        assert (stage3_out / "divergence_curve.png").exists(), "Missing divergence_curve.png"
        assert (stage3_out / "divergence_curve.pdf").exists(), "Missing divergence_curve.pdf"
        assert (stage3_out / "multi_panel_analysis.png").exists(), "Missing multi_panel_analysis.png"
        assert (stage3_out / "multi_panel_analysis.pdf").exists(), "Missing multi_panel_analysis.pdf"
        assert (stage3_out / "divergence_metrics.json").exists(), "Missing divergence_metrics.json"

        print("[PASS] All 5 analysis and figure artifacts generated successfully!")

        # 4. Verify numerical integrity of divergence metrics
        print("\n[Step 4] Verifying metrics contents...")
        metrics = json.loads((stage3_out / "divergence_metrics.json").read_text(encoding="utf-8"))
        assert metrics["total_samples"] == 4
        assert "tvi" in metrics
        assert "divergence_per_layer_all" in metrics
        assert len(metrics["divergence_per_layer_all"]["mean"]) == 5

        print(f"  Auto VIP Layer: {metrics['vip_layer_auto']}")
        print(f"  Avg TVI (All): {metrics['tvi']['all_mean']:.4f}")
        print(f"  Avg TVI (D_T): {metrics['tvi']['dt_mean']:.4f}")
        print(f"  Avg TVI (D_VT): {metrics['tvi']['dvt_mean']:.4f}")
        print(f"  TVI Gap (D_VT - D_T): {metrics['tvi']['tvi_gap']:.4f}")

    print("\n" + "=" * 60)
    print("STAGE 3 INTEGRATION PIPELINE PASSED 100% SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_stage3_end_to_end_pipeline()
