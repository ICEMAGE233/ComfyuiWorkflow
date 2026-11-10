"""
Index-preserving sticker pipeline nodes.

LoadUnprocessedImagesFromFolder
    - reads a folder (sorted), parses the original index from each filename
      (leading digits, e.g. 00001.png -> 1),
    - resumes per-file: skips inputs whose output already exists
      (output/{done_prefix}/{idx:05d}.png), so any range/subset/order resumes,
    - returns (images, masks, indices) lists.

SaveImageIndexed
    - saves image to output/{prefix}/{idx:05d}.png (index-named, so the index
      survives across folder-mediated stages).

UploadToModelScope
    - uploads image to {folder}/Part{idx//1000:02d}/{idx:05d}.png in the dataset,
      idempotent (skips if already present).

Together these guarantee the uploaded numbering matches the original input file,
regardless of folder subset / range / resume.

Restart ComfyUI once after editing this file.
"""

import os
import re
import io
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

import folder_paths


_VALID_EXT = (".png", ".jpg", ".jpeg", ".webp")


def _safe_subdir(root, sub):
    sub = (sub or "").replace("\\", "/").strip("/")
    p = os.path.realpath(os.path.join(root, sub))
    rp = os.path.realpath(root)
    if p != rp and not p.startswith(rp + os.sep):
        raise ValueError(f"Path {sub!r} escapes the allowed directory")
    return p


def _parse_idx(fn):
    """Leading digits of the filename -> int. 00001.png -> 1."""
    m = re.match(r"(\d+)", fn)
    return int(m.group(1)) if m else 0


def _tensor_to_pil(image):
    """[B,H,W,C] or [H,W,C] float tensor -> PIL RGBA/RGB."""
    img = image[0] if image.dim() == 4 else image
    arr = (img.cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr, "RGBA" if arr.shape[-1] == 4 else "RGB")


class LoadUnprocessedImagesFromFolder:
    """Resumable, batched, index-aware folder loader.

    Returns (images, masks, indices). indices are parsed from the input
    filenames so the original identity survives the whole pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "read_from": (["input", "output"], {"default": "input"}),
                "input_folder": ("STRING", {"default": ""}),
                "done_prefix": (
                    "STRING",
                    {"default": "", "tooltip": "output subfolder; an input is 'done' if output/{done_prefix}/{idx:05d}.png exists"},
                ),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "count": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "0 = to end of folder"}),
                "batch_size": ("INT", {"default": 50, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("images", "masks", "indices")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "load"
    CATEGORY = "image"
    DESCRIPTION = "Resumable, batched, index-aware folder loader."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Output depends on filesystem state (which outputs already exist),
        # not captured by the widget cache key -> force re-execution every queue.
        return float("nan")

    def load(self, read_from, input_folder, done_prefix, start_index, count, batch_size):
        root = (
            folder_paths.get_output_directory()
            if read_from == "output"
            else folder_paths.get_input_directory()
        )
        in_dir = _safe_subdir(root, input_folder)
        if not os.path.isdir(in_dir):
            where = f"{read_from}/{input_folder}".strip("/")
            raise ValueError(
                f"Input folder not found: {where!r} (resolved to {in_dir}). "
                f"If this is a later stage, run the previous stage first."
            )
        all_files = sorted(
            f for f in os.listdir(in_dir) if f.lower().endswith(_VALID_EXT)
        )
        if not all_files:
            raise ValueError(f"No images found in input folder: {input_folder!r}")
        all_items = [(f, _parse_idx(f)) for f in all_files]
        # Filter by the parsed INDEX (filename number), not by position in the
        # sorted folder. For FFHQ (idx == position) this is identical; for a
        # cutout/ folder starting at 20000, start_index=20000 correctly hits
        # 20000.png at position 0 instead of 40000.png at position 20000.
        if count <= 0:
            items = [(f, idx) for (f, idx) in all_items if idx >= start_index]
        else:
            items = [(f, idx) for (f, idx) in all_items if start_index <= idx < start_index + count]
        if not items:
            raise ValueError(
                f"Range is empty: start_index={start_index}, count={count}, "
                f"folder has {len(all_items)} images (idx range "
                f"{all_items[0][1]}-{all_items[-1][1]})."
            )
        out_root = folder_paths.get_output_directory()
        done_dir = _safe_subdir(out_root, done_prefix) if done_prefix else out_root
        remaining = [
            (f, idx) for (f, idx) in items
            if not os.path.exists(os.path.join(done_dir, f"{idx:05d}.png"))
        ]
        remaining = remaining[:batch_size]
        logging.info(
            "[ResumeLoader] folder=%r total=%d idx_range=[%d:%s] working=%d remaining=%d batch=%d",
            input_folder, len(all_items), start_index, "+" if count <= 0 else str(start_index + count),
            len(items), len(remaining), batch_size,
        )
        print(
            f"[ResumeLoader] {input_folder!r}: total={len(all_items)}, "
            f"idx>={start_index}" + (f"..{start_index+count}" if count > 0 else "+") + f" working={len(items)}, "
            f"done={len(items) - len(remaining)}, batch={batch_size} -> processing {len(remaining)}"
        )
        if not remaining:
            from comfy_execution.graph import ExecutionBlocker
            return (ExecutionBlocker(None), ExecutionBlocker(None), ExecutionBlocker(None))

        images, masks, indices = [], [], []
        for f, idx in remaining:
            img = Image.open(os.path.join(in_dir, f)).convert("RGBA")
            arr = np.array(img).astype(np.float32) / 255.0
            images.append(torch.from_numpy(arr[:, :, :3])[None,])
            masks.append(torch.from_numpy(arr[:, :, 3])[None,])
            indices.append(idx)
        return (images, masks, indices)


class SaveImageIndexed:
    """Save image to output/{prefix}/{idx:05d}.png (index-named)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "index": ("INT", {"default": 0}),
                "prefix": ("STRING", {"default": "cutout"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "save"
    CATEGORY = "image"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def save(self, image, index, prefix):
        out_dir = os.path.join(folder_paths.get_output_directory(), prefix)
        os.makedirs(out_dir, exist_ok=True)
        _tensor_to_pil(image).save(os.path.join(out_dir, f"{index:05d}.png"))
        return (image,)


class UploadToModelScope:
    """Upload image to {folder}/Part{idx//1000:02d}/{idx:05d}.png in a dataset.

    ASYNC: submits the upload to a background thread pool and returns immediately,
    so image generation proceeds in parallel with uploads. Idempotent (skips files
    already present). The runner script does a final verify/sync pass to catch any
    uploads that failed or were still pending when the run was stopped.
    Token/endpoint come from the widgets (or MODELSCOPE_API_TOKEN env).
    """

    _pool = None
    _pool_lock = threading.Lock()
    _cfg_done = False
    _cfg_lock = threading.Lock()
    _cache = {}
    _cache_lock = threading.Lock()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "index": ("INT", {"default": 0}),
                "folder": (["origin", "illustration"],),
                "repo_id": ("STRING", {"default": "GoatGroup/StickerProject"}),
                "token": ("STRING", {"default": "", "tooltip": "ModelScope ms- token; empty = use MODELSCOPE_API_TOKEN env"}),
                "endpoint": ("STRING", {"default": "https://www.modelscope.ai"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upload"
    CATEGORY = "image"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def _get_pool(cls):
        with cls._pool_lock:
            if cls._pool is None:
                cls._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ms-upload")
            return cls._pool

    @classmethod
    def _ensure_cfg(cls, token, endpoint):
        # config singleton loads at import (server start) with default .cn / no
        # token; override once so all worker threads' HubApi() see .ai + token.
        with cls._cfg_lock:
            if not cls._cfg_done:
                from modelscope_hub.api import HubApi
                api = HubApi()
                api._config.endpoint = endpoint or "https://www.modelscope.ai"
                api._config.token = token
                cls._cfg_done = True

    @classmethod
    def _existing(cls, repo_id, token, endpoint):
        with cls._cache_lock:
            if repo_id in cls._cache:
                return cls._cache[repo_id]
        s = set()
        try:
            from modelscope.hub.api import HubApi as Legacy
            leg = Legacy()
            for f in leg.get_dataset_files(repo_id, token=token, endpoint=endpoint, recursive=True):
                p = f.get("Path") or f.get("Name")
                if p:
                    s.add(p)
        except Exception as e:
            print("[UploadToModelScope] list repo failed:", repr(e)[:200])
        with cls._cache_lock:
            cls._cache[repo_id] = s
            return s

    def upload(self, image, index, folder, repo_id, token, endpoint):
        tok = token or os.environ.get("MODELSCOPE_API_TOKEN")
        if not tok:
            print("[UploadToModelScope] no token -> skip")
            return (image,)
        path = f"{folder}/Part{index // 1000:02d}/{index:05d}.png"
        existing = self._existing(repo_id, tok, endpoint)
        with self._cache_lock:
            if path in existing:
                return (image,)  # already on ModelScope
            existing.add(path)  # claim it so we don't submit duplicates
        # convert to bytes NOW (the tensor may be freed once we return)
        buf = io.BytesIO()
        _tensor_to_pil(image).save(buf, format="PNG")
        data = buf.getvalue()
        self._ensure_cfg(tok, endpoint)
        self._get_pool().submit(self._do_upload, repo_id, path, data, index, folder)
        return (image,)  # return immediately; generation proceeds while upload runs

    @staticmethod
    def _do_upload(repo_id, path, data, index, folder):
        try:
            from modelscope_hub.api import HubApi
            api = HubApi()
            api.upload_file(repo_id, "dataset", data, path, commit_message=f"upload {folder} idx={index}")
            print(f"[UploadToModelScope] uploaded: {path}")
        except Exception as e:
            print(f"[UploadToModelScope] FAILED {path}: {repr(e)[:180]}")


NODE_CLASS_MAPPINGS = {
    "LoadUnprocessedImagesFromFolder": LoadUnprocessedImagesFromFolder,
    "SaveImageIndexed": SaveImageIndexed,
    "UploadToModelScope": UploadToModelScope,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
