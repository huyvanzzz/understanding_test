from pathlib import Path
import sys
import torch

workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from models.ad_vlm.loader import build_ad_vlm_model, parse_yaml_config


def test_decoder_start_extraction():
    config_path = workspace_root / "configs" / "repvit_t5_efficient_mini_kaggle_2gpu_router_vsa_k49.yaml"
    cfg = parse_yaml_config(str(config_path))
    model, tokenizer = build_ad_vlm_model(cfg)
    model.eval()

    prompts = ["Question: What is ahead? Answer:", "Question: Turn left? Answer:"]
    enc = tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    visual_features = torch.randn(2, 6, 49, 512)

    hs_vl, attn_vl = model.extract_decoder_start_representations(
        input_ids=input_ids,
        attention_mask=attention_mask,
        visual_features=visual_features,
        blind_mode=False,
        output_attentions=True,
    )
    hs_l, attn_l = model.extract_decoder_start_representations(
        input_ids=input_ids,
        attention_mask=attention_mask,
        visual_features=None,
        blind_mode=True,
        output_attentions=False,
    )

    num_dec_layers = len(model.text_model.decoder.block) + 1
    assert hs_vl.shape == (2, num_dec_layers, 384), f"Wrong hs_vl shape: {hs_vl.shape}"
    assert hs_l.shape == (2, num_dec_layers, 384), f"Wrong hs_l shape: {hs_l.shape}"
    assert attn_l is None
    assert attn_vl is not None and attn_vl.shape == (2, len(model.text_model.decoder.block)), f"Wrong attn_vl shape: {attn_vl.shape if attn_vl is not None else None}"
    print("[PASS] Decoder start representations verified successfully!")


if __name__ == "__main__":
    test_decoder_start_extraction()
