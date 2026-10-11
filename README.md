# 📌 Attribution & Origin (Nguồn Gốc Dự Án)
This repository is extended and adapted from the official research codebase:  
**[deeplearning-wisc/understanding_lp](https://github.com/deeplearning-wisc/understanding_lp)**  
*(Paper: "Understanding Language Prior of LVLMs by Contrasting Chain-of-Embedding", ICLR 2026)*  
by Lin Long, Changdae Oh, Seongheon Park, and Sharon Li (University of Wisconsin–Madison).
**Core Extensions in this Repository:**
We adapt the Chain-of-Embedding representation divergence methodology ($D_l$, VIP, TVI) to analyze **Multi-View Autonomous Driving Vision-Language Models** (specifically featuring **Question-Guided Token Routers**, **Gated Pooling Attention**, and ultra-compact **RepViT + T5** backbones on the nuScenes / DriveLM dataset).

---

## 🚀 Key Extensions for Autonomous Driving VLMs

1. **Multi-View 6-Camera Fusion Support:**
   - Evaluates multi-camera input fusion modules (Question-Guided Token Router nén 294 tokens xuống $K=49$ tokens, hoặc Gated Pooling Attention baseline).
2. **Dual-Pass Representation Extraction ($VL$ vs $L$):**
   - **Multimodal Pass ($VL$):** Cung cấp đủ 6 góc camera + câu hỏi lái xe $\rightarrow$ Trích xuất vector ẩn từng tầng $h_l^{VL}$.
   - **Blind Language Pass ($L$):** "Bịt mắt" mô hình (chỉ đưa văn bản câu hỏi) $\rightarrow$ Trích xuất vector ẩn $h_l^L$ để đo định kiến ngôn ngữ (Language Prior).
3. **Automated Representation Metrics:**
   - **Layer-wise Divergence Curve ($D_l$):** Đo khoảng cách phân kỳ theo từng tầng mạng giữa trạng thái có ảnh và không ảnh.
   - **Visual Integration Point (VIP):** Tự động phát hiện tầng mạng mà thông tin hình ảnh thực sự bắt đầu được hòa trộn vào ngôn ngữ.
   - **Total Visual Integration (TVI) & TVI Gap:** Định lượng mức độ phụ thuộc vào thị giác ($D_{VT}$) so với mức độ thiên kiến ngôn ngữ ($D_T$).
4. **1-Click Kaggle GPU Automation:**
   - Tối ưu hóa chạy 1-click trên Kaggle GPU (Tesla T4 x2 / P100) thông qua Git clone / Git pull tự động (`notebooks/kaggle_git_ad_analysis.ipynb` và `scripts/kaggle_run_analysis.py`).

---

## 📁 Repository Structure

```text
understanding_lp/
├── analysis/                 # Stage 3: Divergence calculation, VIP & TVI metrics, Matplotlib visualizer
│   ├── divergence_engine.py  # Layer-wise Cosine/Euclidean divergence, auto-VIP, TVI statistics
│   └── visualizer.py         # Publication-grade figures (PNG 300 DPI, PDF 600 DPI)
├── configs/                  # Model configurations (Router K=49, GPA, RepViT, T5-mini)
├── docs/                     # Research reports, verification logs & quickstart guides
│   ├── kaggle_quickstart_guide.md
│   └── repo_analysis_report.md
├── generation/               # Stage 2: Dual-pass extraction (Multimodal VL vs Blind L)
│   └── gen_ad_vlm.py         # Extraction engine supporting feature cache (.memmap) and JSONL
├── models/ad_vlm/            # Stage 1: Self-contained AD-VLM architecture
│   ├── router.py             # Question-Guided Token Router with Visual Self-Attention
│   ├── gpa.py                # Gated Pooling Attention baseline
│   ├── vlm.py                # EfficientVLMForAD model wrapper
│   └── loader.py             # Config parser and weight loader
├── notebooks/                # Kaggle notebooks
│   ├── kaggle_git_ad_analysis.ipynb    # Git-based 1-click execution on Kaggle GPU
│   └── kaggle_ad_analysis_demo.ipynb   # Standalone demo notebook
├── scripts/
│   └── kaggle_run_analysis.py          # 1-Click all-in-one analysis runner
└── tests/                    # Automated verification test suites (Stage 1 to 4)
```

---

## ⚡ Quick Start

### 1. Chạy trên Kaggle GPU (Khuyến nghị)
1. Mở Kaggle $\rightarrow$ **Create New Notebook** $\rightarrow$ **File** $\rightarrow$ **Import Notebook** $\rightarrow$ Chọn `notebooks/kaggle_git_ad_analysis.ipynb`.
2. Bật **GPU T4 x2** và **Internet On**.
3. Điền URL GitHub repo của bạn vào ô số 1 và bấm **Run All**.

### 2. Chạy thử nghiệm trên máy cá nhân (CPU Smoke Test)
```powershell
python scripts/kaggle_run_analysis.py \
    --config configs/repvit_t5_efficient_mini_kaggle_2gpu_router_vsa_k49.yaml \
    --smoke
```

---

## 📚 Citation & Upstream References

Nếu bạn sử dụng mã nguồn hoặc phương pháp luận này, vui lòng trích dẫn bài báo gốc của nhóm tác giả:

```bibtex
@article{long2025understanding,
  title={Understanding Language Prior of LVLMs by Contrasting Chain-of-Embedding},
  author={Long, Lin and Oh, Changdae and Park, Seongheon and Li, Sharon},
  journal={arXiv preprint arXiv:2509.23050},
  year={2025}
}
```
