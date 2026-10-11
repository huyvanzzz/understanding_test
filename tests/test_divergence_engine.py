from __future__ import annotations

import sys
from pathlib import Path

# Add workspace root to sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import torch

from analysis.divergence_engine import (
    calculate_cosine_divergence,
    compute_divergence_metrics,
    compute_tvi_scores,
    detect_vip_layer,
)


def test_divergence_engine():
    print("=" * 60)
    print("TESTING DIVERGENCE & TVI COMPUTATION ENGINE")
    print("=" * 60)

    N, L, D = 4, 5, 384
    # Tạo tensor giả lập: tầng 0 giống nhau hoàn toàn, các tầng sau phân kỳ tăng dần
    hs_vl = torch.randn(N, L, D)
    hs_l = hs_vl.clone()
    for l in range(1, L):
        hs_l[:, l, :] = hs_l[:, l, :] + 0.5 * l * torch.randn(N, D)

    # 1. Kiểm tra tính Cosine Divergence
    div_mat = calculate_cosine_divergence(hs_vl, hs_l)  # [N, L]
    assert div_mat.shape == (N, L), f"Unexpected shape: {div_mat.shape}"
    # Tầng 0 giống hệt nhau -> D_0 gần 0
    assert torch.allclose(div_mat[:, 0], torch.zeros(N), atol=1e-5), "D_0 should be near 0"
    # D_l phải không âm
    assert (div_mat >= -1e-5).all(), "Divergence cannot be negative"

    # 2. Kiểm tra phát hiện tầng VIP
    auto_vip = detect_vip_layer(div_mat)
    assert 1 <= auto_vip < L, f"VIP layer {auto_vip} out of valid range [1, {L-1}]"
    print(f"[PASS] Auto-detected VIP layer: {auto_vip}")

    # 3. Kiểm tra tính TVI scores
    tvi = compute_tvi_scores(div_mat, vip_layer=1)
    assert tvi.shape == (N,), f"TVI shape mismatch: {tvi.shape}"

    # 4. Kiểm tra toàn diện compute_divergence_metrics
    agreements = torch.tensor([True, False, True, False])
    va_weights = torch.rand(N, L - 1)

    metrics = compute_divergence_metrics(
        hidden_states_vl=hs_vl,
        hidden_states_l=hs_l,
        agreement_labels=agreements,
        visual_attention_weights=va_weights,
        vip_layer=1,
    )

    assert metrics["total_samples"] == 4
    assert metrics["dt_count"] == 2
    assert metrics["dvt_count"] == 2
    assert len(metrics["divergence_per_layer_all"]["mean"]) == 5
    assert metrics["divergence_per_layer_all"]["mean"][0] < 1e-4
    assert metrics["vip_layer_used"] == 1
    assert "tvi" in metrics
    assert isinstance(metrics["tvi"]["dt_mean"], float)
    assert isinstance(metrics["tvi"]["dvt_mean"], float)
    assert "spearman" in metrics

    print("[PASS] Full DivergenceEngine metrics verified successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_divergence_engine()
