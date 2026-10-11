from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add workspace root to sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import torch

from analysis.divergence_engine import compute_divergence_metrics
from analysis.visualizer import plot_divergence_curve, plot_multi_panel_analysis


def test_visualizer():
    print("=" * 60)
    print("TESTING PUBLICATION-GRADE VISUALIZER ENGINE")
    print("=" * 60)

    N, L, D = 6, 5, 384
    hs_vl = torch.randn(N, L, D)
    hs_l = hs_vl.clone()
    for l in range(1, L):
        hs_l[:, l, :] = hs_l[:, l, :] + 0.4 * l * torch.randn(N, D)

    agreements = torch.tensor([True, False, True, False, True, False])
    va_weights = torch.rand(N, L - 1)

    metrics = compute_divergence_metrics(
        hidden_states_vl=hs_vl,
        hidden_states_l=hs_l,
        agreement_labels=agreements,
        visual_attention_weights=va_weights,
        vip_layer=1,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        curv_prefix = tmp_path / "divergence_curve"
        multi_prefix = tmp_path / "multi_panel"

        # 1. Test single divergence plot
        res_curv = plot_divergence_curve(
            metrics=metrics,
            title="Router (K=49) Divergence Analysis",
            save_path_prefix=curv_prefix,
            dpi=150,
        )
        assert Path(res_curv["png"]).exists(), "PNG divergence curve not generated!"
        assert Path(res_curv["pdf"]).exists(), "PDF divergence curve not generated!"
        assert Path(res_curv["png"]).stat().st_size > 1000, "PNG file too small!"
        assert Path(res_curv["pdf"]).stat().st_size > 1000, "PDF file too small!"
        print("[PASS] Single divergence curve (PNG + PDF) generated successfully!")

        # 2. Test multi-panel plot
        res_multi = plot_multi_panel_analysis(
            metrics=metrics,
            title="Router Representation & Attention Dynamics",
            save_path_prefix=multi_prefix,
            dpi=150,
        )
        assert Path(res_multi["png"]).exists(), "PNG multi-panel plot not generated!"
        assert Path(res_multi["pdf"]).exists(), "PDF multi-panel plot not generated!"
        assert Path(res_multi["png"]).stat().st_size > 1000, "Multi-panel PNG file too small!"
        assert Path(res_multi["pdf"]).stat().st_size > 1000, "Multi-panel PDF file too small!"
        print("[PASS] Multi-panel analysis plot (PNG + PDF) generated successfully!")

    print("\n" + "=" * 60)
    print("ALL VISUALIZER TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_visualizer()
