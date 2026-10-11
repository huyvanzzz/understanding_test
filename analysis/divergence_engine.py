from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


def calculate_cosine_divergence(
    hs_vl: torch.Tensor,
    hs_l: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute layer-wise cosine divergence between multimodal (VL) and blind (L) hidden states.
    D_l = 1 - CosineSimilarity(h_l^VL, h_l^L)
    
    Args:
        hs_vl: Tensor of shape [N, L, D]
        hs_l:  Tensor of shape [N, L, D]
    Returns:
        Tensor of shape [N, L] with values in [0.0, 2.0]
    """
    assert hs_vl.shape == hs_l.shape, f"Shape mismatch: {hs_vl.shape} vs {hs_l.shape}"
    sim = F.cosine_similarity(hs_vl, hs_l, dim=-1, eps=eps)
    divergence = 1.0 - sim
    return torch.clamp(divergence, min=0.0, max=2.0)


def calculate_euclidean_divergence(
    hs_vl: torch.Tensor,
    hs_l: torch.Tensor,
) -> torch.Tensor:
    """Compute layer-wise Euclidean distance between hidden states."""
    assert hs_vl.shape == hs_l.shape, f"Shape mismatch: {hs_vl.shape} vs {hs_l.shape}"
    return torch.norm(hs_vl - hs_l, dim=-1)


def detect_vip_layer(divergence_matrix: torch.Tensor) -> int:
    """
    Automatically detect the Visual Integration Point (VIP) layer index.
    VIP is defined as the layer where the divergence undergoes the steepest increase (max gradient).
    
    Args:
        divergence_matrix: Tensor [N, L]
    Returns:
        VIP layer index (int in [1, L-1])
    """
    mean_div = torch.mean(divergence_matrix, dim=0)  # [L]
    num_layers = len(mean_div)
    if num_layers <= 2:
        return 1

    diff = torch.diff(mean_div)  # [L-1]
    # Argmax over diff gives index in [0, L-2], so layer index is +1
    best_layer = int(torch.argmax(diff).item()) + 1
    return max(1, min(best_layer, num_layers - 1))


def compute_tvi_scores(
    divergence_matrix: torch.Tensor,
    vip_layer: int,
) -> torch.Tensor:
    """
    Compute Total Visual Integration (TVI) score for each sample.
    TVI(i) = mean_{l >= VIP}(D_l(i))
    
    Args:
        divergence_matrix: Tensor [N, L]
        vip_layer: Starting layer index for integration
    Returns:
        Tensor [N] of per-sample TVI scores
    """
    num_layers = divergence_matrix.shape[1]
    valid_vip = max(0, min(vip_layer, num_layers - 1))
    return torch.mean(divergence_matrix[:, valid_vip:], dim=1)


def compute_divergence_metrics(
    *,
    hidden_states_vl: torch.Tensor,
    hidden_states_l: torch.Tensor,
    agreement_labels: torch.Tensor,
    visual_attention_weights: torch.Tensor | None = None,
    vip_layer: int | str = "auto",
    metric: str = "cosine",
) -> dict[str, Any]:
    """
    Comprehensive representation divergence and TVI diagnostic analysis.
    
    Args:
        hidden_states_vl: Tensor [N, L, D]
        hidden_states_l: Tensor [N, L, D]
        agreement_labels: Tensor [N] (bool, True=D_T, False=D_VT)
        visual_attention_weights: Tensor [N, num_attn_layers] or None
        vip_layer: "auto" or integer layer index
        metric: "cosine" or "euclidean"
    """
    N, L, D = hidden_states_vl.shape
    device = hidden_states_vl.device

    # 1. Compute divergence matrix
    if metric == "cosine":
        div_matrix = calculate_cosine_divergence(hidden_states_vl, hidden_states_l)
    elif metric == "euclidean":
        div_matrix = calculate_euclidean_divergence(hidden_states_vl, hidden_states_l)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'.")

    # 2. Determine VIP layer
    auto_vip = detect_vip_layer(div_matrix)
    if isinstance(vip_layer, str) and vip_layer.lower() == "auto":
        resolved_vip = auto_vip
    else:
        resolved_vip = int(vip_layer)

    resolved_vip = max(1, min(resolved_vip, L - 1))

    # 3. Compute TVI scores
    tvi_scores = compute_tvi_scores(div_matrix, vip_layer=resolved_vip)

    # 4. Partition by Language Prior Agreement (D_T vs D_VT)
    agreements_bool = agreement_labels.bool().cpu()
    dt_mask = agreements_bool
    dvt_mask = ~agreements_bool

    dt_count = int(dt_mask.sum().item())
    dvt_count = int(dvt_mask.sum().item())

    # Layer-wise stats for all samples
    all_mean = torch.mean(div_matrix, dim=0).cpu().tolist()
    all_std = torch.std(div_matrix, dim=0).cpu().tolist()

    # Layer-wise stats for D_T (Language Prior Dominant)
    if dt_count > 0:
        dt_div = div_matrix[dt_mask]
        dt_mean = torch.mean(dt_div, dim=0).cpu().tolist()
        dt_std = torch.std(dt_div, dim=0).cpu().tolist() if dt_count > 1 else [0.0] * L
        dt_tvi_mean = float(torch.mean(tvi_scores[dt_mask]).item())
        dt_tvi_std = float(torch.std(tvi_scores[dt_mask]).item()) if dt_count > 1 else 0.0
    else:
        dt_mean = all_mean
        dt_std = all_std
        dt_tvi_mean = 0.0
        dt_tvi_std = 0.0

    # Layer-wise stats for D_VT (Visual Dependent)
    if dvt_count > 0:
        dvt_div = div_matrix[dvt_mask]
        dvt_mean = torch.mean(dvt_div, dim=0).cpu().tolist()
        dvt_std = torch.std(dvt_div, dim=0).cpu().tolist() if dvt_count > 1 else [0.0] * L
        dvt_tvi_mean = float(torch.mean(tvi_scores[dvt_mask]).item())
        dvt_tvi_std = float(torch.std(tvi_scores[dvt_mask]).item()) if dvt_count > 1 else 0.0
    else:
        dvt_mean = all_mean
        dvt_std = all_std
        dvt_tvi_mean = 0.0
        dvt_tvi_std = 0.0

    all_tvi_mean = float(torch.mean(tvi_scores).item())
    all_tvi_std = float(torch.std(tvi_scores).item()) if N > 1 else 0.0
    tvi_gap = dvt_tvi_mean - dt_tvi_mean

    # 5. Compute Spearman correlation between agreement and TVI
    spearman_stat = {"rho": 0.0, "p_value": 1.0}
    if N > 2 and len(torch.unique(agreement_labels)) > 1:
        try:
            r, p = spearmanr(agreement_labels.float().cpu().numpy(), tvi_scores.cpu().numpy())
            if not np.isnan(r):
                spearman_stat = {"rho": float(r), "p_value": float(p)}
        except Exception:
            pass

    # 6. Process visual attention weights if available
    va_stats: dict[str, Any] = {}
    if visual_attention_weights is not None:
        va_mat = visual_attention_weights.cpu()
        va_stats = {
            "all_mean": torch.mean(va_mat, dim=0).tolist(),
            "all_std": torch.std(va_mat, dim=0).tolist() if N > 1 else [0.0] * va_mat.shape[1],
            "dt_mean": torch.mean(va_mat[dt_mask], dim=0).tolist() if dt_count > 0 else [],
            "dvt_mean": torch.mean(va_mat[dvt_mask], dim=0).tolist() if dvt_count > 0 else [],
        }

    return {
        "total_samples": N,
        "num_layers": L,
        "hidden_dim": D,
        "metric_used": metric,
        "dt_count": dt_count,
        "dvt_count": dvt_count,
        "language_prior_ratio_pct": (dt_count / N) * 100.0 if N > 0 else 0.0,
        "vip_layer_auto": auto_vip,
        "vip_layer_used": resolved_vip,
        "divergence_per_layer_all": {"mean": all_mean, "std": all_std},
        "divergence_per_layer_dt": {"mean": dt_mean, "std": dt_std},
        "divergence_per_layer_dvt": {"mean": dvt_mean, "std": dvt_std},
        "tvi": {
            "all_mean": all_tvi_mean,
            "all_std": all_tvi_std,
            "dt_mean": dt_tvi_mean,
            "dt_std": dt_tvi_std,
            "dvt_mean": dvt_tvi_mean,
            "dvt_std": dvt_tvi_std,
            "tvi_gap": tvi_gap,
        },
        "spearman": spearman_stat,
        "visual_attention_weights": va_stats,
    }
