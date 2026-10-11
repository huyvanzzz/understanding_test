from __future__ import annotations

import json
from pathlib import Path


def create_git_runner_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Autonomous Driving VLM Representation Dynamics & Language Prior Analysis\n",
                    "\n",
                    "**Git-Based 1-Click Execution Runner for Kaggle GPU (Tesla T4 x2 / P100)**\n",
                    "\n",
                    "Notebook này tự động clone/pull mã nguồn mới nhất từ GitHub của bạn về Kaggle, nạp checkpoint và chạy phân tích tương phản biểu diễn ($D_l$, VIP, TVI) theo bài báo ICLR 2026."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =====================================================================\n",
                    "# 1. CẤU HÌNH GITHUB REPOSITORY CỦA BẠN\n",
                    "# =====================================================================\n",
                    "GITHUB_REPO_URL = \"https://github.com/<your-username>/understanding_lp.git\"\n",
                    "GITHUB_BRANCH = \"main\"\n",
                    "\n",
                    "import os\n",
                    "os.environ[\"GITHUB_REPO_URL\"] = GITHUB_REPO_URL\n",
                    "os.environ[\"GITHUB_BRANCH\"] = GITHUB_BRANCH\n",
                    "\n",
                    "# Tự động nạp GitHub Token nếu repo của bạn là Private (tùy chọn)\n",
                    "try:\n",
                    "    from kaggle_secrets import UserSecretsClient\n",
                    "    gh_token = UserSecretsClient().get_secret(\"GITHUB_TOKEN\")\n",
                    "    if gh_token:\n",
                    "        os.environ[\"GITHUB_TOKEN\"] = gh_token\n",
                    "        print(\"[Auth] Đã nạp GITHUB_TOKEN từ Kaggle Secrets!\")\n",
                    "except Exception:\n",
                    "    pass\n",
                    "\n",
                    "print(f\"Target Repo: {GITHUB_REPO_URL} (Branch: {GITHUB_BRANCH})\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "%%bash\n",
                    "set -e\n",
                    "cd /kaggle/working\n",
                    "\n",
                    "# Xử lý token xác thực nếu repo là Private\n",
                    "REPO_URL=\"$GITHUB_REPO_URL\"\n",
                    "if [ -n \"$GITHUB_TOKEN\" ] && [[ \"$REPO_URL\" == https://github.com/* ]]; then\n",
                    "    REPO_URL=\"https://${GITHUB_TOKEN}@${REPO_URL#https://}\"\n",
                    "fi\n",
                    "\n",
                    "# Clone hoặc Pull cập nhật mới nhất từ GitHub\n",
                    "if [ ! -d \"understanding_lp\" ]; then\n",
                    "    echo \"[Git] Đang clone repository từ $GITHUB_REPO_URL ($GITHUB_BRANCH)...\"\n",
                    "    git clone --branch \"$GITHUB_BRANCH\" \"$REPO_URL\" understanding_lp\n",
                    "else\n",
                    "    echo \"[Git] Thư mục đã tồn tại. Đang pull cập nhật mới nhất...\"\n",
                    "    cd understanding_lp\n",
                    "    git checkout \"$GITHUB_BRANCH\"\n",
                    "    git pull --ff-only\n",
                    "fi\n",
                    "\n",
                    "cd /kaggle/working/understanding_lp\n",
                    "echo \"[Dependencies] Cài đặt các thư viện cần thiết...\"\n",
                    "pip install -q pyyaml scipy matplotlib transformers tqdm\n",
                    "echo \"[Environment] Setup môi trường hoàn tất thành công!\""
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =====================================================================\n",
                    "# 2. KIỂM TRA PHẦN CỨNG GPU & PYTORCH TRÊN KAGGLE\n",
                    "# =====================================================================\n",
                    "import torch, platform\n",
                    "\n",
                    "print(f\"Python: {platform.python_version()} | PyTorch: {torch.__version__}\")\n",
                    "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f\"GPU Model: {torch.cuda.get_device_name(0)} (Số lượng: {torch.cuda.device_count()})\")\n",
                    "\n",
                    "!nvidia-smi"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =====================================================================\n",
                    "# 3. CẤU HÌNH & TẢI CHECKPOINT TỪ HUGGING FACE\n",
                    "# =====================================================================\n",
                    "# Nhập thông tin Hugging Face repo và file checkpoint của bạn:\n",
                    "HF_REPO_ID = \"huyvanzzz/ad_vlm_checkpoints\"  # Thay bằng repo HF của bạn\n",
                    "HF_FILENAME = \"finetune_best.pt\"            # Tên file checkpoint trên HF (ví dụ .pt hoặc .pth)\n",
                    "\n",
                    "import os\n",
                    "from huggingface_hub import hf_hub_download\n",
                    "\n",
                    "# Tự động nạp HF_TOKEN từ Kaggle Secrets nếu repo của bạn là Private\n",
                    "try:\n",
                    "    from kaggle_secrets import UserSecretsClient\n",
                    "    hf_token = UserSecretsClient().get_secret(\"HF_TOKEN\")\n",
                    "    if hf_token:\n",
                    "        os.environ[\"HF_TOKEN\"] = hf_token\n",
                    "        print(\"[Auth] Đã nạp HF_TOKEN từ Kaggle Secrets!\")\n",
                    "except Exception:\n",
                    "    pass\n",
                    "\n",
                    "CHECKPOINT_PATH = None\n",
                    "if HF_REPO_ID and HF_FILENAME:\n",
                    "    try:\n",
                    "        print(f\"[Hugging Face] Đang kết nối tải checkpoint: {HF_REPO_ID}/{HF_FILENAME}...\")\n",
                    "        CHECKPOINT_PATH = hf_hub_download(\n",
                    "            repo_id=HF_REPO_ID,\n",
                    "            filename=HF_FILENAME,\n",
                    "            token=os.environ.get(\"HF_TOKEN\")\n",
                    "        )\n",
                    "        print(f\"[Hugging Face] Tải thành công! Lưu tạm tại: {CHECKPOINT_PATH}\")\n",
                    "    except Exception as exc:\n",
                    "        print(f\"[Hugging Face] Thông báo: Chưa tải được từ HF ({exc}).\")\n",
                    "        print(\"[Hugging Face] Script sẽ tự động tìm checkpoint trong /kaggle/input/ hoặc dùng smoke test.\")\n",
                    "        CHECKPOINT_PATH = None"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =====================================================================\n",
                    "# 4. CHẠY PIPELINE PHÂN TÍCH 1-CLICK TRÊN GPU\n",
                    "# =====================================================================\n",
                    "# Mẹo cấu hình:\n",
                    "# - Đặt --max_samples 500: Chạy nhanh ~30-45 giây\n",
                    "# - Xóa --max_samples: Chạy toàn bộ tập dữ liệu test\n",
                    "# - Checkpoint: Tự động dùng file vừa tải từ HF hoặc tự dò tìm\n",
                    "\n",
                    "ckpt_arg = f'--checkpoint \"{CHECKPOINT_PATH}\"' if CHECKPOINT_PATH else \"\"\n",
                    "\n",
                    "!python /kaggle/working/understanding_lp/scripts/kaggle_run_analysis.py \\\n",
                    "    --config /kaggle/working/understanding_lp/configs/repvit_t5_efficient_mini_kaggle_2gpu_router_vsa_k49.yaml \\\n",
                    "    {ckpt_arg} \\\n",
                    "    --output_dir /kaggle/working/outputs/ad_analysis \\\n",
                    "    --batch_size 16 \\\n",
                    "    --max_samples 500 \\\n",
                    "    --device auto"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =====================================================================\n",
                    "# 5. HIỂN THỊ BIỂU ĐỒ KHOA HỌC TRỰC TIẾP TRONG NOTEBOOK\n",
                    "# =====================================================================\n",
                    "from pathlib import Path\n",
                    "from IPython.display import Image, display\n",
                    "\n",
                    "output_dir = Path(\"/kaggle/working/outputs/ad_analysis\")\n",
                    "curve_img = output_dir / \"divergence_curve.png\"\n",
                    "multi_img = output_dir / \"multi_panel_analysis.png\"\n",
                    "\n",
                    "if curve_img.exists():\n",
                    "    print(\"=\" * 65)\n",
                    "    print(\"BIỂU ĐỒ 1: ĐƯỜNG CONG PHÂN KỲ TẦNG D_l (D_T vs D_VT) & TẦNG VIP\")\n",
                    "    print(\"=\" * 65)\n",
                    "    display(Image(filename=str(curve_img), width=850))\n",
                    "\n",
                    "if multi_img.exists():\n",
                    "    print(\"=\" * 65)\n",
                    "    print(\"BIỂU ĐỒ 2: ĐỐI CHIẾU PHÂN KỲ TẦNG & TRỌNG SỐ CHÚ Ý THỊ GIÁC (ATTENTION)\")\n",
                    "    print(\"=\" * 65)\n",
                    "    display(Image(filename=str(multi_img), width=1000))"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =====================================================================\n",
                    "# 6. TỔNG HỢP CHỈ SỐ KHOA HỌC (VIP, TVI GAP, LANGUAGE PRIOR RATIO)\n",
                    "# =====================================================================\n",
                    "import json\n",
                    "\n",
                    "metrics_file = output_dir / \"divergence_metrics.json\"\n",
                    "if metrics_file.exists():\n",
                    "    metrics = json.loads(metrics_file.read_text(encoding=\"utf-8\"))\n",
                    "    print(\"=\" * 65)\n",
                    "    print(\"BÁO CÁO KẾT QUẢ KHOA HỌC (SCIENTIFIC METRICS):\")\n",
                    "    print(\"=\" * 65)\n",
                    "    print(f\"Mô hình phân tích:          {metrics.get('model_name')}\")\n",
                    "    print(f\"Tổng số mẫu đánh giá:      {metrics['total_samples']}\")\n",
                    "    print(f\"Tỷ lệ Language Prior (D_T): {metrics['language_prior_ratio_pct']:.1f}% ({metrics['dt_count']} mẫu)\")\n",
                    "    print(f\"Tỷ lệ Visual Driven (D_VT): {100.0 - metrics['language_prior_ratio_pct']:.1f}% ({metrics['dvt_count']} mẫu)\")\n",
                    "    print(f\"Tầng VIP (Điểm hòa trộn):   Layer {metrics['vip_layer_used']} (Auto-detected: Layer {metrics['vip_layer_auto']})\")\n",
                    "    print(f\"Điểm TVI (Tổng hòa trộn):   {metrics['tvi']['all_mean']:.4f} ± {metrics['tvi']['all_std']:.4f}\")\n",
                    "    print(f\"Khoảng cách TVI Gap:        {metrics['tvi']['tvi_gap']:+.4f} (D_VT - D_T)\")\n",
                    "    print(\"=\" * 65)\n",
                    "\n",
                    "zip_file = output_dir / \"analysis_results_bundle.zip\"\n",
                    "if zip_file.exists():\n",
                    "    print(f\"\\n[OK] Đã đóng gói bundle kết quả: {zip_file} ({zip_file.stat().st_size} bytes)\")\n",
                    "    print(\"=> Bạn có thể tải file ZIP này trực tiếp từ bảng 'Data > Output' bên phải màn hình Kaggle!\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_file = Path(r"D:\AD\understanding_lp\notebooks\kaggle_git_ad_analysis.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Successfully generated: {out_file}")


if __name__ == "__main__":
    create_git_runner_notebook()
