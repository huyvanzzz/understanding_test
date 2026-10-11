from __future__ import annotations

import sys
from pathlib import Path

# Add workspace root to sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import torch
from torch.utils.data import DataLoader

from generation.gen_ad_vlm import ADInferenceDataset, collate_ad_batch


def test_inference_dataset_smoke_and_collate():
    print("=" * 60)
    print("TESTING ADInferenceDataset & BATCH COLLATOR")
    print("=" * 60)

    dataset = ADInferenceDataset(smoke=True, max_samples=6, num_views=6, seq_len=49, vision_dim=512)
    assert len(dataset) == 6, f"Expected 6 samples, got {len(dataset)}"

    sample = dataset[0]
    assert "visual_features" in sample
    assert sample["visual_features"].shape == (6, 49, 512)
    assert isinstance(sample["question"], str)
    assert isinstance(sample["sample_id"], str)

    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_ad_batch)
    batch = next(iter(loader))

    assert batch["visual_features"].shape == (2, 6, 49, 512)
    assert len(batch["questions"]) == 2
    assert len(batch["answers"]) == 2
    assert len(batch["sample_ids"]) == 2

    print("[PASS] ADInferenceDataset smoke and collate verified successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_inference_dataset_smoke_and_collate()
