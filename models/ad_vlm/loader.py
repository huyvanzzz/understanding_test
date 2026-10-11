from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .adapters import DynamicInstructionAdapter
from .gpa import GatedPoolingAttention
from .router import QuestionGuidedTokenRouter
from .vlm import EfficientVLMForAD


class SimpleConfigNamespace:
    """Convenience attribute-based dictionary wrapper for YAML configs."""

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfigNamespace(value))
            elif isinstance(value, list):
                setattr(self, key, [SimpleConfigNamespace(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, SimpleConfigNamespace):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, SimpleConfigNamespace) else item for item in value]
            else:
                result[key] = value
        return result


def parse_yaml_config(config_path_or_data: str | Path | dict[str, Any]) -> SimpleConfigNamespace:
    """Parse a YAML file or raw dict into a SimpleConfigNamespace object."""
    if isinstance(config_path_or_data, dict):
        return SimpleConfigNamespace(config_path_or_data)

    config_path = Path(config_path_or_data)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return SimpleConfigNamespace(data)


def build_ad_vlm_model(cfg: SimpleConfigNamespace) -> tuple[EfficientVLMForAD, Any]:
    """
    Build the EfficientVLMForAD model and tokenizer from configuration without loading checkpoints.
    """
    # 1. Text Model & Tokenizer
    text_model_id = cfg.model.text.model_id
    revision = getattr(cfg.model.text, "revision", None)
    d_model = int(cfg.model.text.d_model)

    tokenizer = AutoTokenizer.from_pretrained(text_model_id, revision=revision)
    text_model = AutoModelForSeq2SeqLM.from_pretrained(text_model_id, revision=revision, attn_implementation="eager")

    # 2. Vision & Dimension configs
    vision_dim = int(cfg.model.vision.output_dim)
    seq_len = int(cfg.model.vision.seq_len)
    num_views = len(getattr(cfg.data, "view_order", ["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"]))
    gpa_hidden_size = int(getattr(getattr(cfg, "training", None), "gpa_hidden_size", 128))

    # 3. Visual Fusion (Router or GPA)
    fusion_name = getattr(cfg.model.fusion, "name", "gpa")
    visual_fusion = None
    if fusion_name == "question_guided_router":
        visual_fusion = QuestionGuidedTokenRouter(
            vision_dim=vision_dim,
            text_dim=d_model,
            seq_len=seq_len,
            num_views=num_views,
            top_k=int(getattr(cfg.model.fusion, "top_k", 49)),
            tau_start=float(getattr(cfg.model.fusion, "tau_start", 2.0)),
            tau_min=float(getattr(cfg.model.fusion, "tau_min", 0.5)),
            diversity_weight=float(getattr(cfg.model.fusion, "diversity_weight", 0.001)),
            eps=float(getattr(cfg.model.fusion, "eps", 1e-8)),
            score_norm=bool(getattr(cfg.model.fusion, "score_norm", True)),
            selected_norm=bool(getattr(cfg.model.fusion, "selected_norm", False)),
            norm_eps=float(getattr(cfg.model.fusion, "norm_eps", 1e-5)),
            visual_self_attention_layers=int(getattr(cfg.model.fusion, "visual_self_attention_layers", 0)),
            visual_self_attention_heads=int(getattr(cfg.model.fusion, "visual_self_attention_heads", 4)),
            visual_self_attention_mlp_ratio=float(getattr(cfg.model.fusion, "visual_self_attention_mlp_ratio", 2.0)),
            visual_self_attention_dropout=float(getattr(cfg.model.fusion, "visual_self_attention_dropout", 0.0)),
            visual_self_attention_residual_scale=float(getattr(cfg.model.fusion, "visual_self_attention_residual_scale", 1e-2)),
        )
    elif fusion_name == "gpa":
        visual_fusion = None  # In EfficientVLMForAD, None triggers faithful GPA
    else:
        raise ValueError(f"Unsupported visual fusion: {fusion_name}")

    # 4. Optional Visual Adapter
    adapter_name = getattr(getattr(cfg.model, "adapter", None), "name", "none")
    visual_adapter = None
    if adapter_name == "minidrive_di":
        visual_adapter = DynamicInstructionAdapter(
            vision_dim=vision_dim,
            text_dim=d_model,
            num_heads=int(getattr(cfg.model.adapter, "num_heads", 4)),
            dropout=float(getattr(cfg.model.adapter, "dropout", 0.0)),
            residual_scale=float(getattr(cfg.model.adapter, "residual_scale", 0.1)),
        )

    # 5. Assemble Model
    model = EfficientVLMForAD(
        text_model=text_model,
        visual_adapter=visual_adapter,
        visual_fusion=visual_fusion,
        vision_dim=vision_dim,
        d_model=d_model,
        seq_len=seq_len,
        gpa_hidden_size=gpa_hidden_size,
    )

    return model, tokenizer


def load_ad_vlm_checkpoint(
    model: EfficientVLMForAD,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> dict[str, Any]:
    """
    Load a trained checkpoint into the EfficientVLMForAD model.
    Handles tied weights, vocabulary resizing mismatches, and sets the model to eval mode.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    device = torch.device(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract state_dict
    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload

    # Handle vocabulary size mismatch (e.g. if '<' token was added during training)
    target_text_model = getattr(model, "text_model", None)
    if target_text_model is not None and hasattr(target_text_model, "get_input_embeddings"):
        current_vocab_size = int(target_text_model.get_input_embeddings().weight.shape[0])

        # Check embedding weight key in state_dict
        embedding_keys = [
            "text_model.shared.weight",
            "text_model.encoder.embed_tokens.weight",
            "text_model.decoder.embed_tokens.weight",
            "shared.weight",
        ]
        ckpt_vocab_size = None
        for key in embedding_keys:
            if key in state_dict:
                ckpt_vocab_size = int(state_dict[key].shape[0])
                break

        if ckpt_vocab_size is not None and ckpt_vocab_size != current_vocab_size:
            target_text_model.resize_token_embeddings(ckpt_vocab_size)

    # Load weights
    load_result = model.load_state_dict(state_dict, strict=strict)

    # Retie text model weights for T5
    if target_text_model is not None and hasattr(target_text_model, "tie_weights"):
        target_text_model.tie_weights()

    # Move to device, set to eval mode, and freeze
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    return {
        "metadata": metadata,
        "missing_keys": getattr(load_result, "missing_keys", []),
        "unexpected_keys": getattr(load_result, "unexpected_keys", []),
    }


def load_ad_vlm_from_config(
    config_path_or_data: str | Path | dict[str, Any],
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> tuple[EfficientVLMForAD, Any, SimpleConfigNamespace]:
    """
    High-level entrypoint:
    Loads configuration, builds model & tokenizer, optionally loads checkpoint,
    and returns (model, tokenizer, config) ready for evaluation.
    """
    cfg = parse_yaml_config(config_path_or_data)
    model, tokenizer = build_ad_vlm_model(cfg)

    if checkpoint_path is not None:
        load_ad_vlm_checkpoint(model, checkpoint_path, device=device)
    else:
        model.to(torch.device(device))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return model, tokenizer, cfg
