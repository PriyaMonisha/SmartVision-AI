# filename: src/utils/helpers.py
# purpose:  Shared utilities — JSON serialization, HuggingFace Hub helpers, timing
# version:  1.0

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── NumpyEncoder ──────────────────────────────────────────────────────────────
# Order is critical (Rule 2):
#   torch.Tensor must come first (PyTorch metrics)
#   np.bool_ BEFORE np.integer (np.bool_ subclasses np.integer in numpy < 2.0)
# Without this every json.dump() on model metrics crashes with TypeError.

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.ndim == 0 else obj.tolist()
        except ImportError:
            pass
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)
    logger.debug(f"Saved JSON: {path}")


def load_json(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── HuggingFace Hub helpers ───────────────────────────────────────────────────
# Rule 37: use HfApi().upload_file() — hf_hub_upload() does not exist
# Always create_repo(exist_ok=True) before first upload

def create_hub_repo(repo_id: str, token: str, private: bool = True) -> None:
    from huggingface_hub import create_repo
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            token=token,
            exist_ok=True,
        )
        logger.info(f"HF Hub repo ready: {repo_id}")
    except Exception as e:
        logger.warning(f"Could not create HF repo: {e}")
        logger.warning("Create manually at https://huggingface.co/new")


def upload_model_to_hub(local_path: Path, filename: str, repo_id: str, token: str) -> None:
    from huggingface_hub import HfApi
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Model file not found: {local_path}")
    size_mb = local_path.stat().st_size / 1e6
    logger.info(f"Uploading {filename} ({size_mb:.0f}MB) to {repo_id}...")
    HfApi().upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    logger.info(f"✅ Uploaded {filename} to {repo_id}")


def download_model_from_hub(
    filename: str,
    local_dir: Path,
    repo_id: str,
    token: str,
) -> Path:
    from huggingface_hub import hf_hub_download
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {filename} from {repo_id}...")
    start = time.time()
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
        token=token,
    )
    elapsed = time.time() - start
    size_mb = Path(path).stat().st_size / 1e6
    logger.info(f"✅ {filename}: {size_mb:.0f}MB in {elapsed:.0f}s")
    return Path(path)


# ── Inference timing ──────────────────────────────────────────────────────────

def benchmark_inference(
    model: Any,
    input_tensor: Any,
    n: int = 100,
    warmup: int = 10,
) -> float:
    """Returns mean inference time in milliseconds over n runs."""
    import torch
    model.eval()
    device = next(model.parameters()).device
    x = input_tensor.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    times = []
    with torch.no_grad():
        for _ in range(n):
            start = time.perf_counter()
            _ = model(x)
            times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times))
