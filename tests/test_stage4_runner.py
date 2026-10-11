from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add workspace root to sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from scripts.kaggle_run_analysis import run_kaggle_analysis_pipeline


def test_kaggle_runner_smoke():
    print("=" * 60)
    print("TESTING KAGGLE AUTOMATION RUNNER")
    print("=" * 60)

    config_path = r"D:\AD\Efficient_VLM_For_Autonomous_Driving_router\configs\repvit_t5_efficient_mini_kaggle_2gpu_router_vsa_k49.yaml"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "kaggle_output"

        res = run_kaggle_analysis_pipeline(
            config_path=config_path,
            checkpoint_path=None,
            cache_dir=None,
            output_dir=out_dir,
            max_samples=4,
            batch_size=2,
            device="cpu",
            smoke=True,
            create_zip=True,
        )

        assert res["success"] is True, "Pipeline run failed!"
        assert (out_dir / "divergence_curve.png").exists(), "divergence_curve.png missing!"
        assert (out_dir / "divergence_curve.pdf").exists(), "divergence_curve.pdf missing!"
        assert (out_dir / "multi_panel_analysis.png").exists(), "multi_panel_analysis.png missing!"
        assert (out_dir / "divergence_metrics.json").exists(), "divergence_metrics.json missing!"
        assert (out_dir / "analysis_results_bundle.zip").exists(), "Zip bundle missing!"

        zip_size = (out_dir / "analysis_results_bundle.zip").stat().st_size
        assert zip_size > 500, f"Zip file size too small: {zip_size} bytes"

        print(f"[PASS] Output bundle created: analysis_results_bundle.zip ({zip_size} bytes)")
        print("[PASS] Kaggle automation runner and zip packaging verified successfully!")
        print("=" * 60)


if __name__ == "__main__":
    test_kaggle_runner_smoke()
