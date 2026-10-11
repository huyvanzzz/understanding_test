from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import torch

from analysis.divergence_engine import compute_divergence_metrics
from analysis.visualizer import plot_divergence_curve, plot_multi_panel_analysis


def run_divergence_analysis(
    *,
    input_dir: str | Path = "outputs/ad_vlm",
    output_dir: str | Path = "outputs/ad_analysis",
    model_name: str = "Router VSA (K=49) + RepViT + T5-Mini",
    vip_layer: int | str = "auto",
    metric: str = "cosine",
    dpi: int = 300,
    target_branch: str | None = None,
) -> dict[str, Any]:
    """
    Core Stage 3 Analysis Pipeline:
    1. Loads extracted representations and agreement labels from input_dir
    2. Computes layer-wise divergence, auto-detects VIP, and calculates TVI scores
    3. Partitions into D_T (Language Prior) vs D_VT (Visual Dependent)
    4. Generates publication-grade single and multi-panel divergence curves (PDF + PNG)
    5. Saves full JSON metrics report
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Auto-detect target_branch from summary.json if not explicitly provided
    if target_branch is None:
        summary_file = in_path / "summary.json"
        if summary_file.exists():
            try:
                summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
                target_branch = summary_data.get("target_branch", "decoder_start")
            except Exception:
                target_branch = "decoder_start"
        else:
            target_branch = "decoder_start"

    print(f"\n[Stage 3] Loading representation tensors from: {in_path} (branch: {target_branch})")
    vl_path = in_path / "hidden_states_vl.pt"
    l_path = in_path / "hidden_states_l.pt"
    agreements_path = in_path / "agreement_labels.pt"
    va_path = in_path / "visual_attention_weights.pt"

    if not vl_path.exists() or not l_path.exists() or not agreements_path.exists():
        raise FileNotFoundError(
            f"Missing required Stage 2 tensors in {in_path}. "
            "Ensure hidden_states_vl.pt, hidden_states_l.pt, and agreement_labels.pt exist."
        )

    hidden_states_vl = torch.load(vl_path, map_location="cpu")
    hidden_states_l = torch.load(l_path, map_location="cpu")
    agreement_labels = torch.load(agreements_path, map_location="cpu")

    visual_attention_weights = None
    if va_path.exists():
        visual_attention_weights = torch.load(va_path, map_location="cpu")
        print(f"[Stage 3] Loaded visual attention weights: shape {list(visual_attention_weights.shape)}")

    print(f"[Stage 3] Loaded VL tensor: {list(hidden_states_vl.shape)}, L tensor: {list(hidden_states_l.shape)}")

    # 1. Compute Divergence and TVI Metrics
    print(f"[Stage 3] Computing layer-wise divergence (metric={metric}, VIP={vip_layer})...")
    metrics = compute_divergence_metrics(
        hidden_states_vl=hidden_states_vl,
        hidden_states_l=hidden_states_l,
        agreement_labels=agreement_labels,
        visual_attention_weights=visual_attention_weights,
        vip_layer=vip_layer,
        metric=metric,
    )
    metrics["model_name"] = model_name
    metrics["target_branch"] = target_branch

    # 2. Render Publication-Grade Visualizations
    print("[Stage 3] Rendering publication-grade divergence figures...")
    curve_prefix = out_path / "divergence_curve"
    multi_prefix = out_path / "multi_panel_analysis"

    curve_paths = plot_divergence_curve(
        metrics=metrics,
        save_path_prefix=curve_prefix,
        dpi=dpi,
        target_branch=target_branch,
        model_name=model_name,
    )

    multi_paths = plot_multi_panel_analysis(
        metrics=metrics,
        save_path_prefix=multi_prefix,
        dpi=dpi,
        target_branch=target_branch,
        model_name=model_name,
    )

    metrics["output_files"] = {
        "divergence_curve_png": str(curve_paths["png"]),
        "divergence_curve_pdf": str(curve_paths["pdf"]),
        "multi_panel_analysis_png": str(multi_paths["png"]),
        "multi_panel_analysis_pdf": str(multi_paths["pdf"]),
    }

    # 3. Export JSON Metrics Report
    metrics_file = out_path / "divergence_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 4. Display Formatted Console Summary
    tvi_info = metrics["tvi"]
    vip_used = metrics["vip_layer_used"]
    vip_auto = metrics["vip_layer_auto"]
    total = metrics["total_samples"]
    dt_c = metrics["dt_count"]
    dvt_c = metrics["dvt_count"]
    lp_ratio = metrics["language_prior_ratio_pct"]

    print("\n" + "=" * 65)
    print(f"STAGE 3 ANALYSIS REPORT: {model_name.upper()}")
    print("=" * 65)
    print(f"Total Evaluated Samples:     {total}")
    print(f"  - D_T  (Language Prior):   {dt_c} ({lp_ratio:.1f}%)")
    print(f"  - D_VT (Visual Dependent): {dvt_c} ({100.0 - lp_ratio:.1f}%)")
    print("-" * 65)
    print(f"VIP (Visual Integration Point): Layer {vip_used} (Auto-detected: Layer {vip_auto})")
    print(f"Total Visual Integration (TVI) Scores:")
    print(f"  - All Samples TVI:         {tvi_info['all_mean']:.4f} +/- {tvi_info['all_std']:.4f}")
    print(f"  - D_T  (Prior Dominant):   {tvi_info['dt_mean']:.4f} +/- {tvi_info['dt_std']:.4f}")
    print(f"  - D_VT (Vision Driven):    {tvi_info['dvt_mean']:.4f} +/- {tvi_info['dvt_std']:.4f}")
    print(f"  - TVI Integration Gap:     {tvi_info['tvi_gap']:+.4f} (D_VT - D_T)")
    if metrics["spearman"]["p_value"] < 1.0:
        print(f"Spearman Correlation (rho):   {metrics['spearman']['rho']:.4f} (p={metrics['spearman']['p_value']:.4e})")
    print("-" * 65)
    print("Generated Artifacts:")
    print(f"  [PNG] {curve_paths['png']}")
    print(f"  [PDF] {curve_paths['pdf']}")
    print(f"  [PNG] {multi_paths['png']}")
    print(f"  [PDF] {multi_paths['pdf']}")
    print(f"  [JSON] {metrics_file}")
    print("=" * 65 + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Divergence Calculation & Publication Visualization Engine")
    parser.add_argument("--input_dir", type=str, default="outputs/ad_vlm", help="Directory containing Stage 2 output tensors")
    parser.add_argument("--output_dir", type=str, default="outputs/ad_analysis", help="Directory to save plots and JSON metrics")
    parser.add_argument("--model_name", type=str, default="Router VSA (K=49) + RepViT + T5-Mini", help="Model label for figures")
    parser.add_argument("--vip", type=str, default="auto", help="VIP layer index ('auto' or integer)")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"], help="Divergence metric")
    parser.add_argument("--dpi", type=int, default=300, help="DPI resolution for output PNG figures")
    parser.add_argument(
        "--target_branch",
        type=str,
        default=None,
        choices=["decoder_start", "encoder_last"],
        help="Target extraction point ('decoder_start' or 'encoder_last'). Auto-detected from summary.json if omitted.",
    )

    args = parser.parse_args()

    vip_val: int | str = args.vip
    if args.vip.isdigit():
        vip_val = int(args.vip)

    run_divergence_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        vip_layer=vip_val,
        metric=args.metric,
        dpi=args.dpi,
        target_branch=args.target_branch,
    )


if __name__ == "__main__":
    main()
