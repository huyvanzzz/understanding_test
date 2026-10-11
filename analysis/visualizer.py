from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Ensure headless backend for non-GUI environments (CLI / Kaggle / Servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _apply_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def _layer_labels(num_layers: int) -> list[str]:
    if num_layers == 5:
        return ["L0 (Embed)", "L1 (Block 0)", "L2 (Block 1)", "L3 (Block 2)", "L4 (Final)"]
    labels = []
    for i in range(num_layers):
        if i == 0:
            labels.append("L0 (Embed)")
        elif i == num_layers - 1:
            labels.append(f"L{i} (Final)")
        else:
            labels.append(f"L{i}")
    return labels


def plot_divergence_curve(
    metrics: dict[str, Any],
    title: str | None = None,
    save_path_prefix: str | Path = "outputs/ad_analysis/divergence_curve",
    figsize: tuple[int, int] = (11, 6),
    dpi: int = 300,
    target_branch: str | None = None,
    model_name: str | None = None,
) -> dict[str, str]:
    """
    Generate publication-grade primary divergence curve:
    - Compares D_T (Language Prior Dominant) vs D_VT (Visual Dependent)
    - Shaded ±0.5 sigma standard deviation band
    - Vertical dashed VIP marker
    - Exports both vector PDF and raster PNG
    """
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if target_branch is None:
        target_branch = metrics.get("target_branch", "decoder_start")
    is_decoder = (target_branch == "decoder_start")

    if title is None:
        prefix_title = f"{model_name}\n" if model_name else ""
        branch_desc = "T5 Decoder Start-Token" if is_decoder else "T5 Encoder Last-Token"
        title = f"{prefix_title}{branch_desc} Divergence Curve ($D_l$)"

    num_layers = metrics.get("num_layers", len(metrics["divergence_per_layer_all"]["mean"]))
    layers = list(range(num_layers))
    x_labels = _layer_labels(num_layers)

    dt_count = metrics.get("dt_count", 0)
    dvt_count = metrics.get("dvt_count", 0)
    vip_layer = metrics.get("vip_layer_used", 1)

    tvi_info = metrics.get("tvi", {})
    dt_tvi = tvi_info.get("dt_mean", 0.0)
    dvt_tvi = tvi_info.get("dvt_mean", 0.0)

    # 1. Plot D_T (Language Prior Dominant)
    if dt_count > 0:
        dt_mean = np.array(metrics["divergence_per_layer_dt"]["mean"])
        dt_std = np.array(metrics["divergence_per_layer_dt"]["std"])
        color_dt = "#1f77b4"  # Royal Navy Blue
        ax.plot(
            layers,
            dt_mean,
            marker="o",
            linestyle="-",
            color=color_dt,
            linewidth=3.0,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=f"$D_{{T}}$ (Avg TVI={dt_tvi:.3f}) [n={dt_count}]",
            zorder=3,
        )
        ax.fill_between(
            layers,
            dt_mean - 0.5 * dt_std,
            dt_mean + 0.5 * dt_std,
            color=color_dt,
            alpha=0.20,
            edgecolor=color_dt,
            linewidth=0.5,
            zorder=2,
        )

    # 2. Plot D_VT (Visual Dependent)
    if dvt_count > 0:
        dvt_mean = np.array(metrics["divergence_per_layer_dvt"]["mean"])
        dvt_std = np.array(metrics["divergence_per_layer_dvt"]["std"])
        color_dvt = "#d62728"  # Crimson Coral Red
        ax.plot(
            layers,
            dvt_mean,
            marker="s",
            linestyle="-",
            color=color_dvt,
            linewidth=3.0,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=f"$D_{{VT}}$ (Avg TVI={dvt_tvi:.3f}) [n={dvt_count}]",
            zorder=3,
        )
        ax.fill_between(
            layers,
            dvt_mean - 0.5 * dvt_std,
            dvt_mean + 0.5 * dvt_std,
            color=color_dvt,
            alpha=0.20,
            edgecolor=color_dvt,
            linewidth=0.5,
            zorder=2,
        )

    # 3. Always plot Overall population mean if both or only one partition is present
    all_mean = np.array(metrics["divergence_per_layer_all"]["mean"])
    all_std = np.array(metrics["divergence_per_layer_all"]["std"])
    ax.plot(
        layers,
        all_mean,
        marker="^",
        linestyle=":",
        color="#7b1fa2",
        linewidth=1.8,
        markersize=6,
        alpha=0.7,
        label=f"All Samples (Avg TVI={tvi_info.get('all_mean', 0.0):.3f})",
        zorder=2,
    )

    # 4. VIP Layer vertical dashed line
    if 0 <= vip_layer < num_layers:
        ax.axvline(
            x=vip_layer,
            color="#2c3e50",
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            label=f"VIP (Layer {vip_layer})",
            zorder=1,
        )

    # Styling Axes
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15, color="#2c3e50")
    ax_xlabel = "T5 Decoder Layer Depth" if is_decoder else "T5 Encoder Layer Depth"
    ax.set_xlabel(ax_xlabel, fontsize=13, fontweight="600", labelpad=10, color="#34495e")
    ax_ylabel = (
        r"Divergence $D_l = 1 - \mathrm{Cosine}(h_{l, \mathrm{dec}}^{VL}, h_{l, \mathrm{dec}}^L)$"
        if is_decoder
        else r"Divergence $D_l = 1 - \mathrm{Cosine}(h_l^{VL}, h_l^L)$"
    )
    ax.set_ylabel(ax_ylabel, fontsize=13, fontweight="600", labelpad=10, color="#34495e")

    ax.set_xticks(layers)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight="500")
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, color="#bdc3c7")

    # Legend
    ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=11,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#dee2e6",
        borderpad=0.6,
    )

    plt.tight_layout()

    prefix = Path(save_path_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = prefix.with_suffix(".png")
    pdf_path = prefix.with_suffix(".pdf")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {"png": str(png_path), "pdf": str(pdf_path)}


def plot_multi_panel_analysis(
    metrics: dict[str, Any],
    title: str | None = None,
    save_path_prefix: str | Path = "outputs/ad_analysis/multi_panel_analysis",
    figsize: tuple[int, int] = (16, 6),
    dpi: int = 300,
    target_branch: str | None = None,
    model_name: str | None = None,
) -> dict[str, str]:
    """
    Generate side-by-side 2-panel publication figure:
    - Panel A: Layer-wise Representation Divergence D_l
    - Panel B: Visual Attention / Cross-Attention Weight
    """
    _apply_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    if target_branch is None:
        target_branch = metrics.get("target_branch", "decoder_start")
    is_decoder = (target_branch == "decoder_start")

    if title is None:
        model_part = f"{model_name}: " if model_name else ""
        branch_desc = "T5 Decoder Start-Token" if is_decoder else "T5 Encoder Last-Token"
        title = f"{model_part}{branch_desc} Representation & Attention Analysis"

    num_layers = metrics.get("num_layers", len(metrics["divergence_per_layer_all"]["mean"]))
    layers = list(range(num_layers))
    x_labels = _layer_labels(num_layers)

    dt_count = metrics.get("dt_count", 0)
    dvt_count = metrics.get("dvt_count", 0)
    vip_layer = metrics.get("vip_layer_used", 1)
    tvi_info = metrics.get("tvi", {})

    # ==================== PANEL A: DIVERGENCE ====================
    if dt_count > 0:
        dt_mean = np.array(metrics["divergence_per_layer_dt"]["mean"])
        dt_std = np.array(metrics["divergence_per_layer_dt"]["std"])
        ax1.plot(
            layers,
            dt_mean,
            marker="o",
            linestyle="-",
            color="#1f77b4",
            linewidth=2.5,
            markersize=7,
            label=f"$D_{{T}}$ (TVI={tvi_info.get('dt_mean', 0.0):.3f})",
        )
        ax1.fill_between(layers, dt_mean - 0.5 * dt_std, dt_mean + 0.5 * dt_std, color="#1f77b4", alpha=0.18)

    if dvt_count > 0:
        dvt_mean = np.array(metrics["divergence_per_layer_dvt"]["mean"])
        dvt_std = np.array(metrics["divergence_per_layer_dvt"]["std"])
        ax1.plot(
            layers,
            dvt_mean,
            marker="s",
            linestyle="-",
            color="#d62728",
            linewidth=2.5,
            markersize=7,
            label=f"$D_{{VT}}$ (TVI={tvi_info.get('dvt_mean', 0.0):.3f})",
        )
        ax1.fill_between(layers, dvt_mean - 0.5 * dvt_std, dvt_mean + 0.5 * dvt_std, color="#d62728", alpha=0.18)

    if 0 <= vip_layer < num_layers:
        ax1.axvline(x=vip_layer, color="#2c3e50", linestyle="--", linewidth=1.5, alpha=0.8, label=f"VIP (L{vip_layer})")

    ax1.set_title("Panel A: Layer-wise Divergence $D_l$", fontsize=14, fontweight="bold", pad=12)
    ax1_xlabel = "T5 Decoder Layer Depth" if is_decoder else "T5 Encoder Layer Depth"
    ax1.set_xlabel(ax1_xlabel, fontsize=12, fontweight="600", labelpad=8)
    ax1_ylabel = (
        r"$D_l = 1 - \mathrm{Cosine}(h_{l, \mathrm{dec}}^{VL}, h_{l, \mathrm{dec}}^L)$"
        if is_decoder
        else r"$D_l = 1 - \mathrm{Cosine}(h_l^{VL}, h_l^L)$"
    )
    ax1.set_ylabel(ax1_ylabel, fontsize=12, fontweight="600", labelpad=8)
    ax1.set_xticks(layers)
    ax1.set_xticklabels(x_labels, fontsize=10)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.35)

    # ==================== PANEL B: VISUAL / CROSS ATTENTION ====================
    va_data = metrics.get("visual_attention_weights", {})
    if va_data and "all_mean" in va_data and len(va_data["all_mean"]) > 0:
        num_attn_blocks = len(va_data["all_mean"])
        attn_x = list(range(1, num_attn_blocks + 1))
        attn_labels = [f"Block {i}" for i in attn_x]

        dt_label = r"$D_{T}$ Cross-Attention" if is_decoder else r"$D_{T}$ Visual Attention"
        dvt_label = r"$D_{VT}$ Cross-Attention" if is_decoder else r"$D_{VT}$ Visual Attention"
        all_label = "All Samples Cross-Attention" if is_decoder else "All Samples Attention"

        if dt_count > 0 and len(va_data.get("dt_mean", [])) > 0:
            ax2.plot(
                attn_x,
                va_data["dt_mean"],
                marker="o",
                color="#1f77b4",
                linewidth=2.5,
                markersize=7,
                label=dt_label,
            )
        if dvt_count > 0 and len(va_data.get("dvt_mean", [])) > 0:
            ax2.plot(
                attn_x,
                va_data["dvt_mean"],
                marker="s",
                color="#d62728",
                linewidth=2.5,
                markersize=7,
                label=dvt_label,
            )

        ax2.plot(
            attn_x,
            va_data["all_mean"],
            marker="^",
            linestyle=":",
            color="#7b1fa2",
            linewidth=1.8,
            label=all_label,
        )
        ax2.set_xticks(attn_x)
        ax2.set_xticklabels(attn_labels, fontsize=10)
    else:
        ax2.text(0.5, 0.5, "Attention Tensor Not Provided", ha="center", va="center", fontsize=12, color="#7f8c8d")

    panel_b_title = "Panel B: Decoder Cross-Attention to Vision" if is_decoder else "Panel B: Query-to-Vision Attention Weight"
    ax2.set_title(panel_b_title, fontsize=14, fontweight="bold", pad=12)
    ax2_xlabel = "T5 Decoder Transformer Block" if is_decoder else "T5 Transformer Block"
    ax2.set_xlabel(ax2_xlabel, fontsize=12, fontweight="600", labelpad=8)
    ax2_ylabel = (
        r"Cross-Attention Weight $\sum \alpha_{pad \to vis}$"
        if is_decoder
        else r"Attention Weight $\sum \alpha_{last \to vis}$"
    )
    ax2.set_ylabel(ax2_ylabel, fontsize=12, fontweight="600", labelpad=8)
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.35)

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02, color="#2c3e50")
    plt.tight_layout()

    prefix = Path(save_path_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = prefix.with_suffix(".png")
    pdf_path = prefix.with_suffix(".pdf")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {"png": str(png_path), "pdf": str(pdf_path)}
