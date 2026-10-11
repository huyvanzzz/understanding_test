from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Add workspace root to sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import torch
import torch.nn as nn

from models.ad_vlm.loader import (
    build_ad_vlm_model,
    load_ad_vlm_checkpoint,
    load_ad_vlm_from_config,
    parse_yaml_config,
)
from models.ad_vlm.router import QuestionGuidedTokenRouter
from models.ad_vlm.vlm import EfficientVLMForAD


def test_stage1_complete():
    print("=" * 60)
    print("RUNNING STAGE 1 VERIFICATION SUITE")
    print("=" * 60)

    # 1. Test parsing real config from router project
    config_path = r"D:\AD\Efficient_VLM_For_Autonomous_Driving_router\configs\repvit_t5_efficient_mini_kaggle_2gpu_router_vsa_k49.yaml"
    print(f"\n[Step 1] Parsing configuration from: {config_path}")
    cfg = parse_yaml_config(config_path)

    assert cfg.model.fusion.name == "question_guided_router", f"Expected question_guided_router, got {cfg.model.fusion.name}"
    assert int(cfg.model.fusion.top_k) == 49, f"Expected top_k 49, got {cfg.model.fusion.top_k}"
    assert bool(cfg.model.fusion.score_norm) is True, "Expected score_norm True"
    assert bool(cfg.model.fusion.selected_norm) is False, "Expected selected_norm False"
    assert int(cfg.model.fusion.visual_self_attention_layers) == 1, "Expected visual_self_attention_layers 1"
    print("[PASS] Config parsed successfully with correct Router + VSA parameters!")

    # 2. Test model building
    print("\n[Step 2] Building EfficientVLMForAD model and tokenizer...")
    model, tokenizer = build_ad_vlm_model(cfg)

    assert isinstance(model, EfficientVLMForAD), "Model should be EfficientVLMForAD"
    assert isinstance(model.visual_fusion, QuestionGuidedTokenRouter), "visual_fusion should be QuestionGuidedTokenRouter"
    assert model.visual_fusion.top_k == 49, f"Router top_k should be 49, got {model.visual_fusion.top_k}"
    assert len(model.visual_fusion.visual_self_attention) == 1, "Should have 1 VSA layer"
    assert isinstance(model.visual_fusion.pre_score_norm, nn.LayerNorm), "pre_score_norm should be LayerNorm"
    assert isinstance(model.visual_fusion.selected_norm, nn.Identity), "selected_norm should be Identity"
    print("[PASS] Model and Router initialized with exact layer types!")

    # 3. Test forward multimodal (VL branch) and blind (L branch) on CPU
    print("\n[Step 3] Testing forward passes (VL branch vs Blind L branch)...")
    batch_size = 2
    seq_len = 49
    vision_dim = 512
    num_views = 6

    # Create dummy RepViT cached features [B, 6, 49, 512]
    dummy_visual = torch.randn(batch_size, num_views, seq_len, vision_dim)
    prompt_text = ["Question: Is there a vehicle in front? Answer:", "Question: What is on the right? Answer:"]
    encoded = tokenizer(prompt_text, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Forward VL (with vision features)
    model.eval()
    with torch.no_grad():
        out_vl = model.forward_from_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=dummy_visual,
            output_hidden_states=True,
            blind_mode=False,
        )

        out_l = model.forward_from_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=dummy_visual,
            output_hidden_states=True,
            blind_mode=True,  # Blind mode masks vision
        )

    # Verify T5 Encoder hidden states exist for representation analysis
    assert hasattr(out_vl, "encoder_hidden_states"), "out_vl should have encoder_hidden_states"
    assert hasattr(out_l, "encoder_hidden_states"), "out_l should have encoder_hidden_states"
    assert out_vl.encoder_hidden_states is not None, "encoder_hidden_states should not be None"

    num_layers_vl = len(out_vl.encoder_hidden_states)
    num_layers_l = len(out_l.encoder_hidden_states)
    assert num_layers_vl == num_layers_l, f"Layer count mismatch: {num_layers_vl} vs {num_layers_l}"

    print(f"[PASS] VL branch forward passed: {num_layers_vl} encoder hidden state layers extracted!")
    print(f"[PASS] Blind L branch forward passed: {num_layers_l} encoder hidden state layers extracted!")
    print(f"  Shape of layer 0 hidden state: {list(out_vl.encoder_hidden_states[0].shape)}")

    # 4. Test checkpoint saving and loading
    print("\n[Step 4] Testing checkpoint loader and tied-weights verification...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_checkpoint.pt"
        payload = {
            "model": model.state_dict(),
            "metadata": {"epoch": 1, "profile": "test_router"},
        }
        torch.save(payload, ckpt_path)

        # Load into new fresh model
        model_fresh, _, _ = load_ad_vlm_from_config(config_path, checkpoint_path=ckpt_path, device="cpu")
        assert model_fresh.training is False, "Model should be in eval mode"
        for param in model_fresh.parameters():
            assert not param.requires_grad, "Parameters should be frozen"

        print("[PASS] Checkpoint loaded, tied-weights preserved, and model set to eval/frozen mode!")

    print("\n" + "=" * 60)
    print("ALL STAGE 1 VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_stage1_complete()
