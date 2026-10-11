from pathlib import Path
import sys
import torch

workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from generation.gen_ad_vlm import extract_dual_pass_representations


def test_gen_target_branch_option(tmp_path):
    config_path = workspace_root / "configs" / "repvit_t5_efficient_mini_kaggle_2gpu_router_vsa_k49.yaml"
    out_dir = tmp_path / "decoder_extract_out"

    summary = extract_dual_pass_representations(
        config_path=str(config_path),
        output_dir=str(out_dir),
        max_samples=2,
        batch_size=2,
        smoke=True,
        target_branch="decoder_start",
    )

    assert summary["target_branch"] == "decoder_start", f"Wrong target_branch in summary: {summary.get('target_branch')}"
    hs_vl = torch.load(out_dir / "hidden_states_vl.pt", weights_only=True)
    # T5-mini decoder has 4 blocks + 1 embed = 5 layers
    assert hs_vl.shape == (2, 5, 384), f"Wrong hidden_states_vl shape: {hs_vl.shape}"
    print("[PASS] gen_ad_vlm target_branch option verified successfully!")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_gen_target_branch_option(Path(td))
