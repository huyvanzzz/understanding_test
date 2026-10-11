from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

CAMERA_ORDER = ["Front", "Front-Left", "Front-Right", "Back", "Back-Left", "Back-Right"]

NUSCENES_CAMERA_TO_CANONICAL = {
    "CAM_FRONT": "Front",
    "CAM_FRONT_LEFT": "Front-Left",
    "CAM_FRONT_RIGHT": "Front-Right",
    "CAM_BACK": "Back",
    "CAM_BACK_LEFT": "Back-Left",
    "CAM_BACK_RIGHT": "Back-Right",
}


def extract_cameras(row: dict[str, Any]) -> dict[str, str]:
    source = row.get("cameras", row) if isinstance(row.get("cameras"), dict) else row
    cameras: dict[str, str] = {}
    for cam in CAMERA_ORDER:
        if cam in source:
            cameras[cam] = str(source[cam])
    for raw_k, canon_k in NUSCENES_CAMERA_TO_CANONICAL.items():
        if raw_k in source:
            cameras[canon_k] = str(source[raw_k])
    return cameras


def parse_drivelm_json(json_path: str | Path, split: str = "test") -> list[dict[str, Any]]:
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if isinstance(item, list) and len(item) == 2 and isinstance(item[0], dict) and isinstance(item[1], dict):
            row = {"Q": item[0].get("Q"), "A": item[0].get("A"), "cameras": item[1]}
        elif isinstance(item, dict):
            row = item
        else:
            continue

        q = row.get("question") or row.get("Q")
        a = row.get("answer") or row.get("A")
        if not q or not a:
            continue

        cams = extract_cameras(row)
        if not cams:
            continue

        records.append(
            {
                "sample_id": str(row.get("sample_id") or row.get("id") or f"{split}-{idx}"),
                "eval_id": idx,
                "question": str(q),
                "answer": str(a),
                "camera_paths": cams,
                "split": split,
            }
        )
    return records


def download_and_prepare_drivelm(
    repo_id: str = "minhdang0901/drivelm-nuscenes-v1-0",
    split: str = "test",
    output_file: str | Path = "data/test_qa.jsonl",
    token: str | None = None,
) -> Path:
    """
    Download lightweight QA JSON annotations (~5MB) from Hugging Face and format as jsonl.
    Requires HF_TOKEN if repo is gated or private.
    """
    token = token or os.environ.get("HF_TOKEN")
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[DriveLM] Downloading QA annotations from '{repo_id}' (split={split})...")
    local_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["*.json"],
            token=token,
        )
    )

    # Find JSON file for the requested split
    files = list(local_dir.rglob("*.json"))
    matched = None
    for f in files:
        fname = f.name.lower()
        if split in fname and ("multi_frame" in fname or "frame" in fname):
            matched = f
            break
    if not matched:
        for f in files:
            if split in f.name.lower():
                matched = f
                break

    if not matched:
        raise FileNotFoundError(f"Could not find '{split}' JSON file in {local_dir}. Found: {[f.name for f in files]}")

    print(f"[DriveLM] Parsing annotations from: {matched.name}...")
    records = parse_drivelm_json(matched, split=split)
    print(f"[DriveLM] Successfully parsed {len(records)} QA records.")

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[DriveLM] Saved QA dataset to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download and prepare DriveLM QA annotations")
    parser.add_argument("--repo_id", type=str, default="minhdang0901/drivelm-nuscenes-v1-0", help="Hugging Face Dataset repo ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--output", type=str, default="data/test_qa.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    download_and_prepare_drivelm(repo_id=args.repo_id, split=args.split, output_file=args.output)


if __name__ == "__main__":
    main()
