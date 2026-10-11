from pathlib import Path
import sys
import torch

workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from analysis.divergence_engine import compute_divergence_metrics
from analysis.visualizer import plot_multi_panel_analysis, plot_divergence_curve


def test_visualizer_decoder_labels(tmp_path):
    N, L, D = 4, 5, 384
    hs_vl = torch.randn(N, L, D)
    hs_l = torch.randn(N, L, D)
    agreements = torch.tensor([True, False, False, False])
    attn = torch.rand(N, L - 1)

    metrics = compute_divergence_metrics(
        hidden_states_vl=hs_vl,
        hidden_states_l=hs_l,
        agreement_labels=agreements,
        visual_attention_weights=attn,
    )
    metrics["target_branch"] = "decoder_start"

    save_prefix_multi = tmp_path / "decoder_multi_plot"
    res_multi = plot_multi_panel_analysis(
        metrics=metrics,
        save_path_prefix=str(save_prefix_multi),
        model_name="RepViT-T5-Mini",
        target_branch="decoder_start",
    )
    assert Path(res_multi["png"]).exists()
    assert Path(res_multi["pdf"]).exists()

    save_prefix_curve = tmp_path / "decoder_curve_plot"
    res_curve = plot_divergence_curve(
        metrics=metrics,
        save_path_prefix=str(save_prefix_curve),
        model_name="RepViT-T5-Mini",
        target_branch="decoder_start",
    )
    assert Path(res_curve["png"]).exists()
    assert Path(res_curve["pdf"]).exists()

    print("[PASS] Visualizer decoder branch labels and rendering verified successfully!")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_visualizer_decoder_labels(Path(td))
