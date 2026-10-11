from __future__ import annotations

import sys
from pathlib import Path

# Add workspace root to sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import torch

from models.ad_vlm.loader import build_ad_vlm_model, parse_yaml_config


def test_dynamic_token_indexing_and_attentions():
    print("=" * 60)
    print("TESTING DYNAMIC TOKEN INDEXING & ATTENTION EXTRACTION")
    print("=" * 60)

    config_path = r"D:\AD\Efficient_VLM_For_Autonomous_Driving_router\configs\repvit_t5_efficient_mini_kaggle_2gpu_router_vsa_k49.yaml"
    cfg = parse_yaml_config(config_path)
    model, tokenizer = build_ad_vlm_model(cfg)
    model.eval()

    # 1. Hai prompt có độ dài chênh lệch lớn
    prompts = [
        "Question: Stop? Answer:",  # Ngắn (5 tokens)
        "Question: Is there a large commercial truck parked on the left shoulder of the road? Answer:",  # Dài (18 tokens)
    ]
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    visual_features = torch.randn(2, 6, 49, 512)

    with torch.no_grad():
        hs_vl, attn_vl = model.extract_last_token_representations(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            blind_mode=False,
            output_attentions=True,
        )
        hs_l, _ = model.extract_last_token_representations(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=None,
            blind_mode=True,
            output_attentions=False,
        )

    # 2. Kiểm tra shape tensor: T5-Efficient-Mini có 4 encoder blocks + initial embeddings = 5 layers
    num_layers = len(model.text_model.encoder.block) + 1
    assert hs_vl.shape == (2, num_layers, 384), f"Unexpected hs_vl shape: {hs_vl.shape}"
    assert hs_l.shape == (2, num_layers, 384), f"Unexpected hs_l shape: {hs_l.shape}"
    assert attn_vl is not None and attn_vl.shape == (2, num_layers - 1), f"Unexpected attn_vl shape: {attn_vl.shape}"

    # 3. Kiểm tra tính bất biến (Invariance test):
    # Khi chạy prompt 0 một mình (không có padding), vector biểu diễn phải khớp chính xác với khi chạy trong batch!
    enc_single = tokenizer([prompts[0]], padding=False, return_tensors="pt")
    with torch.no_grad():
        hs_vl_single, _ = model.extract_last_token_representations(
            input_ids=enc_single["input_ids"],
            attention_mask=enc_single["attention_mask"],
            visual_features=visual_features[:1],
            blind_mode=False,
        )

    diff = torch.norm(hs_vl[0] - hs_vl_single[0]).item()
    cos_sim = torch.nn.functional.cosine_similarity(hs_vl[0], hs_vl_single[0], dim=-1).mean().item()
    print(f"Norm difference between batched and unpadded extraction: {diff:.6e}")
    print(f"Mean cosine similarity between batched and unpadded: {cos_sim:.6f}")
    assert cos_sim > 0.9999, f"Cosine similarity too low! Got {cos_sim}"
    assert diff < 1e-3, f"Dynamic indexing mismatch with unpadded single sample! Diff: {diff}"

    print("[PASS] Dynamic token indexing and visual attention extraction verified successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_dynamic_token_indexing_and_attentions()
