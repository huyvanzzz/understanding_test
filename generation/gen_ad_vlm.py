from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.ad_vlm.loader import load_ad_vlm_from_config


def format_prompt(question: str) -> str:
    """Format driving VQA question according to EM-VLM4AD prompt template."""
    return f"Question: {question.strip()} Answer:"


def normalize_text(text: str) -> str:
    """Normalize text for consistent agreement comparison."""
    return text.strip().lower().rstrip(".")


class ADInferenceDataset(Dataset):
    """
    Flexible dataset loader supporting:
    1. Synthetic smoke data (for fast CPU verification without huge datasets)
    2. Real feature cache (.memmap) with index.json and manifest.json
    3. JSON/JSONL dataset files with camera_paths or sample_ids
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        data_file: str | Path | None = None,
        max_samples: int | None = None,
        smoke: bool = False,
        num_views: int = 6,
        seq_len: int = 49,
        vision_dim: int = 512,
    ) -> None:
        self.samples: list[dict[str, Any]] = []
        self.memmap_features = None

        if smoke or cache_dir is None or not Path(cache_dir).exists():
            # Synthetic / smoke testing mode
            count = max_samples or 10
            for i in range(count):
                self.samples.append(
                    {
                        "sample_id": f"smoke_sample_{i}",
                        "eval_id": i,
                        "question": f"Is the road ahead clear for ego vehicle at camera view {i % num_views}?",
                        "answer": "Yes, proceed safely." if i % 2 == 0 else "No, obstacle in front.",
                        "visual_features": torch.randn(num_views, seq_len, vision_dim),
                    }
                )
            return

        cache_path = Path(cache_dir)
        manifest_file = cache_path / "manifest.json"
        index_file = cache_path / "index.json"
        memmap_file = cache_path / "features.float16.memmap"

        if manifest_file.exists() and index_file.exists() and memmap_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            index = json.loads(index_file.read_text(encoding="utf-8"))
            shape = tuple(manifest.get("shape", []))

            self.memmap_features = np.memmap(
                memmap_file,
                mode="r",
                dtype=np.float16,
                shape=shape,
            )

            # If user provided a specific data json/jsonl file
            if data_file and Path(data_file).exists():
                records: list[dict[str, Any]] = []
                data_path = Path(data_file)
                if data_path.suffix == ".jsonl":
                    with open(data_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                records.append(json.loads(line))
                else:
                    records = json.loads(data_path.read_text(encoding="utf-8"))

                for rec in records:
                    cam_paths = rec.get("camera_paths")
                    cam_key = "|".join(cam_paths[k] for k in sorted(cam_paths)) if isinstance(cam_paths, dict) else None
                    sample_key = rec.get("sample_id") or rec.get("id") or rec.get("eval_id")

                    feature_idx = None
                    if cam_key and cam_key in index:
                        feature_idx = index[cam_key]
                    elif sample_key and sample_key in index:
                        feature_idx = index[sample_key]
                    elif str(sample_key) in index:
                        feature_idx = index[str(sample_key)]

                    if feature_idx is not None:
                        self.samples.append(
                            {
                                "sample_id": str(sample_key or len(self.samples)),
                                "eval_id": rec.get("eval_id", len(self.samples)),
                                "question": rec.get("question", ""),
                                "answer": rec.get("answer", ""),
                                "feature_idx": feature_idx,
                            }
                        )
            else:
                # Use index keys directly if no data_file specified
                for key, f_idx in index.items():
                    self.samples.append(
                        {
                            "sample_id": str(key),
                            "eval_id": len(self.samples),
                            "question": "Describe the status of traffic around the ego vehicle.",
                            "answer": "Normal driving conditions.",
                            "feature_idx": f_idx,
                        }
                    )

            if max_samples is not None:
                self.samples = self.samples[:max_samples]
        else:
            raise FileNotFoundError(
                f"Missing cache files in {cache_dir}. Expected manifest.json, index.json, features.float16.memmap."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        if "visual_features" in item:
            visual_tensor = item["visual_features"]
        else:
            f_idx = item["feature_idx"]
            visual_tensor = torch.from_numpy(np.array(self.memmap_features[f_idx], dtype=np.float32))

        return {
            "sample_id": item["sample_id"],
            "eval_id": item.get("eval_id", idx),
            "question": item["question"],
            "answer": item.get("answer", ""),
            "visual_features": visual_tensor,
        }


def collate_ad_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "eval_ids": [item["eval_id"] for item in batch],
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "visual_features": torch.stack([item["visual_features"] for item in batch]),
    }


def extract_dual_pass_representations(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    data_file: str | Path | None = None,
    output_dir: str | Path = "outputs/ad_vlm",
    max_samples: int | None = None,
    batch_size: int = 4,
    device: str = "auto",
    smoke: bool = False,
    max_new_tokens: int = 32,
    output_attentions: bool = True,
    target_branch: str = "decoder_start",
) -> dict[str, Any]:
    """
    Core Dual-Pass Extraction Engine:
    1. Loads model & tokenizer using loader.py (with eager attention for attention maps).
    2. Runs batch inference across 2 branches:
       - VL Branch: Input = [Visual Features + Text Prompt]
       - L Branch:  Input = [Blind Text Prompt only]
    3. Extracts layer-wise last-token hidden states from T5 Encoder using dynamic length indexing.
    4. Extracts visual attention weights from query token to visual tokens.
    5. Generates predictions to partition samples into D_T (same prediction) vs D_VT (different prediction).
    6. Saves standardized PyTorch tensors and JSON metadata.
    """
    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    print(f"[Device] Using compute device: {target_device}")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load model and tokenizer
    print(f"[Model] Loading model from config: {config_path}")
    if checkpoint_path:
        print(f"[Checkpoint] Loading weights from: {checkpoint_path}")
    model, tokenizer, cfg = load_ad_vlm_from_config(config_path, checkpoint_path=checkpoint_path, device=target_device)
    model.eval()

    # 2. Build dataset and dataloader
    print("[Dataset] Initializing inference dataset...")
    num_views = len(getattr(cfg.data, "view_order", ["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"]))
    seq_len = int(cfg.model.vision.seq_len)
    vision_dim = int(cfg.model.vision.output_dim)

    dataset = ADInferenceDataset(
        cache_dir=cache_dir,
        data_file=data_file,
        max_samples=max_samples,
        smoke=smoke,
        num_views=num_views,
        seq_len=seq_len,
        vision_dim=vision_dim,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ad_batch)
    print(f"[Dataset] Ready to process {len(dataset)} samples in {len(dataloader)} batches.")

    all_hidden_states_vl: list[torch.Tensor] = []
    all_hidden_states_l: list[torch.Tensor] = []
    all_visual_attentions: list[torch.Tensor] = []
    all_agreement_labels: list[bool] = []
    metadata_records: list[dict[str, Any]] = []

    # 3. Batch Extraction Loop
    for batch in tqdm(dataloader, desc="Extracting Dual-Pass Hidden States"):
        questions = batch["questions"]
        answers_gt = batch["answers"]
        sample_ids = batch["sample_ids"]
        eval_ids = batch["eval_ids"]
        visual_features = batch["visual_features"].to(target_device)

        prompts = [format_prompt(q) for q in questions]
        encoded = tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(target_device)
        attention_mask = encoded["attention_mask"].to(target_device)

        with torch.no_grad():
            # --- Pass 1: VL Branch (Multimodal) ---
            hs_vl, attn_vl = model.extract_representations(
                target_branch=target_branch,
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=visual_features,
                blind_mode=False,
                output_attentions=output_attentions,
            )

            # Generate VL prediction text
            gen_ids_vl = model.generate_from_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=visual_features,
                max_new_tokens=max_new_tokens,
                blind_mode=False,
            )
            preds_vl = tokenizer.batch_decode(gen_ids_vl, skip_special_tokens=True)

            # --- Pass 2: L Branch (Blind / Pure Text) ---
            hs_l, _ = model.extract_representations(
                target_branch=target_branch,
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=None,
                blind_mode=True,
                output_attentions=False,
            )

            # Generate L prediction text
            gen_ids_l = model.generate_from_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=None,
                max_new_tokens=max_new_tokens,
                blind_mode=True,
            )
            preds_l = tokenizer.batch_decode(gen_ids_l, skip_special_tokens=True)

        all_hidden_states_vl.append(hs_vl)
        all_hidden_states_l.append(hs_l)
        if attn_vl is not None:
            all_visual_attentions.append(attn_vl)

        # Compare predictions to classify D_T vs D_VT
        for s_id, e_id, q, gt, p_vl, p_l in zip(sample_ids, eval_ids, questions, answers_gt, preds_vl, preds_l):
            is_same = normalize_text(p_vl) == normalize_text(p_l)
            all_agreement_labels.append(is_same)
            metadata_records.append(
                {
                    "sample_id": s_id,
                    "eval_id": e_id,
                    "question": q,
                    "ground_truth": gt,
                    "pred_vl": p_vl,
                    "pred_l": p_l,
                    "is_same_prediction": is_same,
                }
            )

    # 4. Concatenate and Save Results
    final_hs_vl = torch.cat(all_hidden_states_vl, dim=0)  # [N, num_layers, D]
    final_hs_l = torch.cat(all_hidden_states_l, dim=0)  # [N, num_layers, D]
    final_agreements = torch.tensor(all_agreement_labels, dtype=torch.bool)

    assert final_hs_vl.shape == final_hs_l.shape, f"Shape mismatch: {final_hs_vl.shape} vs {final_hs_l.shape}"
    assert len(final_hs_vl) == len(final_agreements) == len(metadata_records)

    # Save standardized tensor files
    torch.save(final_hs_vl, out_path / "hidden_states_vl.pt")
    torch.save(final_hs_l, out_path / "hidden_states_l.pt")
    torch.save(final_agreements, out_path / "agreement_labels.pt")

    if all_visual_attentions:
        final_attns = torch.cat(all_visual_attentions, dim=0)  # [N, num_transformer_layers]
        torch.save(final_attns, out_path / "visual_attention_weights.pt")

    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2, ensure_ascii=False)

    total_samples = len(metadata_records)
    same_count = int(final_agreements.sum().item())
    diff_count = total_samples - same_count
    dt_ratio = (same_count / total_samples) * 100 if total_samples > 0 else 0.0

    summary = {
        "target_branch": target_branch,
        "total_samples": total_samples,
        "d_t_count": same_count,
        "d_vt_count": diff_count,
        "language_prior_ratio_pct": dt_ratio,
        "tensor_shape": list(final_hs_vl.shape),
        "visual_attention_shape": list(final_attns.shape) if all_visual_attentions else None,
        "output_dir": str(out_path),
    }

    with open(out_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Target branch: {target_branch}")
    print(f"Total samples: {total_samples}")
    print(f"D_T samples (Language Prior Dominant - Same): {same_count} ({dt_ratio:.1f}%)")
    print(f"D_VT samples (Visual Dependent - Different): {diff_count} ({100.0 - dt_ratio:.1f}%)")
    print(f"Tensor shape (N, Layers, Dim): {list(final_hs_vl.shape)}")
    if all_visual_attentions:
        print(f"Visual attention shape (N, Layers): {list(final_attns.shape)}")
    print(f"Output files saved to: {out_path}")
    print("=" * 60)

    return summary


# Alias for backward compatibility
extract_ad_vlm_representations = extract_dual_pass_representations


def main():
    parser = argparse.ArgumentParser(description="Dual-Pass Hidden State Extractor for Autonomous Driving VLM")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt/.pth (optional)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Path to feature cache directory")
    parser.add_argument("--data_file", type=str, default=None, help="Path to prepared records JSON/JSONL")
    parser.add_argument("--output_dir", type=str, default="outputs/ad_vlm", help="Directory to save extracted tensors")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for extraction")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--smoke", action="store_true", help="Run with synthetic smoke data on CPU")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max generated tokens for prediction")
    parser.add_argument("--no_attentions", action="store_true", help="Disable visual attention tracking")
    parser.add_argument(
        "--target_branch",
        type=str,
        default="decoder_start",
        choices=["decoder_start", "encoder_last"],
        help="Extraction point: 'decoder_start' (T5 Decoder <pad> token) or 'encoder_last' (Encoder last token).",
    )

    args = parser.parse_args()

    extract_dual_pass_representations(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache_dir,
        data_file=args.data_file,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        smoke=args.smoke,
        max_new_tokens=args.max_new_tokens,
        output_attentions=not args.no_attentions,
        target_branch=args.target_branch,
    )


if __name__ == "__main__":
    main()
