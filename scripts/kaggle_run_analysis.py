from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import torch

from generation.gen_ad_vlm import extract_dual_pass_representations
from models.ad_vlm.loader import SimpleConfigNamespace, parse_yaml_config
from plot_divergences_ad import run_divergence_analysis


def resolve_environment() -> dict[str, Any]:
    """Inspect compute environment, GPU availability, and Kaggle secrets."""
    is_kaggle = Path("/kaggle/working").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    cuda_avail = torch.cuda.is_available()

    env_info = {
        "is_kaggle": is_kaggle,
        "cuda_available": cuda_avail,
        "device_count": torch.cuda.device_count() if cuda_avail else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cwd": str(Path.cwd()),
    }

    # Attempt to load HF_TOKEN from Kaggle Secrets if running on Kaggle
    if is_kaggle and not os.environ.get("HF_TOKEN"):
        try:
            from kaggle_secrets import UserSecretsClient

            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
                env_info["hf_token_loaded"] = True
        except Exception:
            env_info["hf_token_loaded"] = False
    else:
        env_info["hf_token_loaded"] = bool(os.environ.get("HF_TOKEN"))

    return env_info


def auto_discover_checkpoint(explicit_path: str | Path | None = None) -> Path | None:
    """
    Auto-search for checkpoint file in Kaggle Input Datasets, local output dirs,
    or download directly from Hugging Face Hub (e.g. 'username/repo/file.pt').
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p

        # Check if explicit_path has Hugging Face repo format (e.g. 'username/repo_name/file.pt')
        raw_str = str(explicit_path).replace("\\", "/").strip()
        parts = [part for part in raw_str.split("/") if part]
        if len(parts) >= 3:
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = "/".join(parts[2:])
            try:
                from huggingface_hub import hf_hub_download

                token = os.environ.get("HF_TOKEN")
                print(f"[Checkpoint] Downloading from Hugging Face Hub: {repo_id} ({filename})...")
                downloaded_file = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
                print(f"[Checkpoint] Successfully downloaded to: {downloaded_file}")
                return Path(downloaded_file)
            except Exception as exc:
                print(f"[Checkpoint] Failed downloading from Hugging Face Hub: {exc}")

    # Search candidates in Kaggle and local directories
    search_patterns = [
        "/kaggle/input/**/*.pt",
        "/kaggle/input/**/*.pth",
        "/kaggle/working/**/*.pt",
        "outputs/**/checkpoints/*.pt",
        "outputs/**/*.pt",
    ]

    for pattern in search_patterns:
        matches = glob.glob(pattern, recursive=True)
        for m in matches:
            p = Path(m)
            # Prioritize finetune or align checkpoints
            if any(k in p.name.lower() for k in ("finetune", "align", "checkpoint", "best", "latest")):
                return p

    return None


def auto_discover_cache_dir(explicit_path: str | Path | None = None) -> Path | None:
    """Auto-search for feature cache directory containing features.float16.memmap."""
    if explicit_path and Path(explicit_path).exists():
        return Path(explicit_path)

    search_patterns = [
        "/kaggle/input/**/features.float16.memmap",
        "outputs/**/cache/features.float16.memmap",
    ]

    for pattern in search_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return Path(matches[0]).parent

    return None


def create_zip_bundle(source_dir: Path, zip_dest: Path) -> Path:
    """Package analysis figures, metrics, and summary into a single downloadable zip file."""
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in source_dir.iterdir():
            if f.is_file() and f.name != zip_dest.name and not f.name.endswith(".pt"):
                zf.write(f, arcname=f.name)
    return zip_dest


def derive_model_display_name(config_path_or_cfg: str | Path | SimpleConfigNamespace | None) -> str:
    """
    Automatically derive a clean, publication-ready display label from model config.
    Used for figure titles in Stage 3 divergence plots.
    """
    if config_path_or_cfg is None:
        return "Autonomous Driving VLM"
    try:
        if isinstance(config_path_or_cfg, (str, Path)):
            cfg = parse_yaml_config(config_path_or_cfg)
        else:
            cfg = config_path_or_cfg

        model_cfg = getattr(cfg, "model", None)
        if model_cfg is None:
            return "Autonomous Driving VLM"

        fusion_cfg = getattr(model_cfg, "fusion", None)
        fusion_type = str(getattr(fusion_cfg, "name", "fusion")).lower() if fusion_cfg else "fusion"
        top_k = getattr(fusion_cfg, "top_k", None) if fusion_cfg else None

        vision_cfg = getattr(model_cfg, "vision", None)
        vision_name = str(getattr(vision_cfg, "name", "RepViT")).upper() if vision_cfg else "RepViT"

        text_cfg = getattr(model_cfg, "text", None)
        text_id = str(getattr(text_cfg, "model_id", "T5")) if text_cfg else "T5"
        text_name = "T5" if "t5" in text_id.lower() else text_id.split("/")[-1]

        if "router" in fusion_type:
            k_suffix = f" (K={top_k})" if top_k else ""
            return f"Question-Guided Router{k_suffix} + {vision_name} + {text_name}"
        elif "gpa" in fusion_type:
            return f"GPA + {vision_name} + {text_name}"
        else:
            profile = getattr(model_cfg, "profile", None)
            if profile:
                return str(profile)
            return f"{fusion_type.title()} + {vision_name} + {text_name}"
    except Exception:
        return "Autonomous Driving VLM"


def run_kaggle_analysis_pipeline(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    data_file: str | Path | None = None,
    output_dir: str | Path = "/kaggle/working/outputs/ad_analysis",
    max_samples: int | None = None,
    batch_size: int = 16,
    device: str = "auto",
    smoke: bool = False,
    vip_layer: int | str = "auto",
    model_name: str | None = None,
    create_zip: bool = True,
) -> dict[str, Any]:
    """
    All-in-one Autonomous Driving Representation Analysis Runner:
    1. Checks GPU and environment
    2. Auto-resolves checkpoint and cache paths
    3. Executes Stage 2 Dual-Pass Extraction on GPU
    4. Executes Stage 3 Divergence, VIP & TVI Visualization
    5. Packages results into a 1-click downloadable zip file
    """
    resolved_model_name = model_name or derive_model_display_name(config_path)

    print("\n" + "=" * 65)
    print("STARTING 1-CLICK AUTONOMOUS DRIVING ANALYSIS RUNNER")
    print(f"Model Display Label: {resolved_model_name}")
    print("=" * 65)

    env = resolve_environment()
    print(f"[Environment] Running on Kaggle: {env['is_kaggle']}")
    print(f"[Environment] Python: {env['python_version']}, PyTorch: {env['torch_version']}")
    if env["cuda_available"]:
        print(f"[Environment] GPU Detected: {env['device_name']} (Count: {env['device_count']})")
    else:
        print("[Environment] No GPU detected. Running on CPU mode.")

    out_path = Path(output_dir)
    raw_tensor_dir = out_path / "raw_tensors"
    out_path.mkdir(parents=True, exist_ok=True)
    raw_tensor_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve Paths
    ckpt = auto_discover_checkpoint(checkpoint_path)
    if ckpt:
        print(f"[Checkpoint] Using checkpoint: {ckpt}")
    else:
        if not smoke:
            print("[Checkpoint] No checkpoint specified or discovered. Proceeding with initial model weights.")

    resolved_cache = auto_discover_cache_dir(cache_dir)
    if resolved_cache:
        print(f"[Feature Cache] Using cache directory: {resolved_cache}")
    elif not smoke:
        print("[Feature Cache] No cache directory found. (Use --cache_dir or --smoke)")

    # 2. Determine compute device & batch size
    if device == "auto":
        target_device = "cuda" if env["cuda_available"] else "cpu"
    else:
        target_device = device

    actual_batch_size = batch_size
    if target_device == "cpu" and actual_batch_size > 4:
        actual_batch_size = 4  # Limit CPU batch size to prevent sluggishness

    print(f"[Compute] Target device: {target_device}, Batch size: {actual_batch_size}")

    # 3. Stage 2: Dual-Pass Hidden State & Attention Extraction
    print("\n" + "-" * 65)
    print("STEP 1: DUAL-PASS EXTRACTION (Multimodal VL vs Blind L)")
    print("-" * 65)

    extraction_summary = extract_dual_pass_representations(
        config_path=config_path,
        checkpoint_path=ckpt,
        cache_dir=resolved_cache,
        data_file=data_file,
        output_dir=raw_tensor_dir,
        max_samples=max_samples,
        batch_size=actual_batch_size,
        device=target_device,
        smoke=smoke,
        output_attentions=True,
    )

    if target_device == "cuda":
        torch.cuda.empty_cache()

    # 4. Stage 3: Divergence Calculation, VIP & TVI Visualization
    print("\n" + "-" * 65)
    print("STEP 2: REPRESENTATION DIVERGENCE & PUBLICATION VISUALIZATION")
    print("-" * 65)

    analysis_metrics = run_divergence_analysis(
        input_dir=raw_tensor_dir,
        output_dir=out_path,
        model_name=resolved_model_name,
        vip_layer=vip_layer,
        metric="cosine",
        dpi=300,
    )

    # 5. Create Markdown Summary
    summary_md_path = out_path / "analysis_summary.md"
    tvi_info = analysis_metrics["tvi"]
    md_content = rf"""# Autonomous Driving VLM Representation Dynamics Report

**Model:** {resolved_model_name}  
**Total Evaluated Samples:** {analysis_metrics['total_samples']}  
**Language Prior Ratio ($D_T$):** {analysis_metrics['language_prior_ratio_pct']:.1f}% ({analysis_metrics['dt_count']} samples)  
**Visual Dependent Ratio ($D_{{VT}}$):** {100.0 - analysis_metrics['language_prior_ratio_pct']:.1f}% ({analysis_metrics['dvt_count']} samples)  

---

### Visual Integration Point (VIP) & TVI Scores

* **VIP Layer Index:** Layer {analysis_metrics['vip_layer_used']} (Auto-detected: Layer {analysis_metrics['vip_layer_auto']})
* **Total Visual Integration (TVI) - All Samples:** {tvi_info['all_mean']:.4f} $\pm$ {tvi_info['all_std']:.4f}
* **TVI - Language Prior Dominant ($D_T$):** {tvi_info['dt_mean']:.4f} $\pm$ {tvi_info['dt_std']:.4f}
* **TVI - Visual Dependent ($D_{{VT}}$):** {tvi_info['dvt_mean']:.4f} $\pm$ {tvi_info['dvt_std']:.4f}
* **TVI Integration Gap ($\Delta$ TVI):** {tvi_info['tvi_gap']:+.4f}

---

### Generated Artifacts
* `divergence_curve.png` / `divergence_curve.pdf`: Layer-wise divergence curve with $\pm 0.5\sigma$ shaded intervals.
* `multi_panel_analysis.png` / `multi_panel_analysis.pdf`: 2-Panel figure comparing divergence and attention weights.
* `divergence_metrics.json`: Full numerical layer-wise statistics.
"""
    summary_md_path.write_text(md_content, encoding="utf-8")

    # 6. Package into ZIP file
    zip_path = None
    if create_zip:
        zip_path = out_path / "analysis_results_bundle.zip"
        create_zip_bundle(out_path, zip_path)
        print(f"\n[Packaging] Analysis bundle archived into: {zip_path}")

    print("\n" + "=" * 65)
    print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Results directory: {out_path}")
    if zip_path and zip_path.exists():
        print(f"Downloadable ZIP:   {zip_path} ({zip_path.stat().st_size} bytes)")
    print("=" * 65 + "\n")

    return {
        "success": True,
        "output_dir": str(out_path),
        "zip_path": str(zip_path) if zip_path else None,
        "metrics": analysis_metrics,
        "extraction_summary": extraction_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="1-Click Kaggle Autonomous Driving Representation Analysis Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to model YAML configuration")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint .pt (optional, auto-searched)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Path to feature cache directory (optional, auto-searched)")
    parser.add_argument("--data_file", type=str, default=None, help="Path to dataset JSON/JSONL records")
    parser.add_argument("--output_dir", type=str, default="outputs/ad_analysis", help="Output directory for plots & metrics")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to analyze")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--smoke", action="store_true", help="Run with synthetic smoke data")
    parser.add_argument("--vip", type=str, default="auto", help="VIP layer index ('auto' or int)")
    parser.add_argument("--model_name", type=str, default=None, help="Label for plots (optional, auto-derived from config)")
    parser.add_argument("--no_zip", action="store_true", help="Do not package into zip file")

    args = parser.parse_args()

    vip_val: int | str = args.vip
    if args.vip.isdigit():
        vip_val = int(args.vip)

    run_kaggle_analysis_pipeline(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache_dir,
        data_file=args.data_file,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        smoke=args.smoke,
        vip_layer=vip_val,
        model_name=args.model_name,
        create_zip=not args.no_zip,
    )


if __name__ == "__main__":
    main()
