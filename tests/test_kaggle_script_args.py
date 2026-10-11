from pathlib import Path
import subprocess
import sys

workspace_root = Path(__file__).resolve().parent.parent


def test_kaggle_run_analysis_help():
    cmd = [sys.executable, str(workspace_root / "scripts" / "kaggle_run_analysis.py"), "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "--target_branch" in res.stdout, "Expected --target_branch in kaggle_run_analysis.py --help"
    print("[PASS] kaggle_run_analysis --target_branch option verified successfully!")


if __name__ == "__main__":
    test_kaggle_run_analysis_help()
