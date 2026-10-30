"""
Resume-safe image folder loader.

Adds one node: LoadUnprocessedImagesFromFolder

It reads an input folder (sorted, deterministic order) and returns ONLY the
images whose corresponding output has not yet been produced. Output progress is
inferred by counting existing saved files that match a given filename prefix
(the same string you put in the downstream SaveImage "filename_prefix").

Why count instead of name-match: ComfyUI's SaveImage numbers outputs
sequentially (prefix_00001_.png, prefix_00002_.png, ...) in processing order,
and ComfyUI iterates an output list sequentially, so "N output files exist"
<=> "the first N inputs (in sorted order) are done". Sorting the inputs makes
that mapping deterministic; the stock LoadImageDataSetFromFolder does NOT sort,
which is why naive "skip the first N" resume is unreliable.

Drop-in replacement for LoadImageDataSetFromFolder at the head of any staged
pipeline. On interruption, just re-queue: finished images are skipped
automatically and the run resumes from the first unfinished one.

Restart ComfyUI once after adding this file so the node is registered.
"""

import os
import logging

import numpy as np
import torch
from PIL import Image

import folder_paths


_VALID_EXT = (".png", ".jpg", ".jpeg", ".webp")


def _safe_subdir(root, sub):
    """Join root + sub and ensure the result stays under root (no traversal)."""
    sub = (sub or "").replace("\\", "/").strip("/")
    p = os.path.realpath(os.path.join(root, sub))
    rp = os.path.realpath(root)
    if p != rp and not p.startswith(rp + os.sep):
        raise ValueError(f"Path {sub!r} escapes the allowed directory")
    return p


def _split_prefix(prefix):
    """'cutout/image' -> ('cutout', 'image'); 'image' -> ('', 'image')."""
    parts = (prefix or "").replace("\\", "/").strip("/").split("/")
    name = parts[-1] if parts else ""
    return "/".join(parts[:-1]), name


class LoadUnprocessedImagesFromFolder:
    """Load only the input images that have not yet produced an output file.

    Inputs:
      read_from     - 'input' or 'output': which ComfyUI directory the
                      input_folder lives under. Stage A reads source photos
                      from input/; stages B and C read the previous stage's
                      saved results from output/, so no manual file moving
                      is needed between stages.
      input_folder  - subfolder under the chosen directory.
      done_prefix   - the EXACT string used in the downstream SaveImage
                      "filename_prefix" for this stage (e.g. 'cutout/image').
                      Existing files matching {name}_*.png under
                      output/{subfolder} are counted as "done".

    Outputs (both are lists -> downstream runs once per image):
      images - RGB image per remaining input.
      masks  - alpha channel per remaining input (all-ones for RGB inputs;
               useful when the next step composites the subject onto a
               background).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "read_from": (
                    ["input", "output"],
                    {"default": "input", "tooltip": "Which ComfyUI directory input_folder is under. Stages B/C use 'output' to read the previous stage's saved results."},
                ),
                "input_folder": (
                    "STRING",
                    {"default": "", "tooltip": "Subfolder under the chosen directory to read images from."},
                ),
                "done_prefix": (
                    "STRING",
                    {
                        "default": "image",
                        "tooltip": "Same string as this stage's SaveImage filename_prefix. Existing output files matching {name}_*.png are treated as done. Use a UNIQUE prefix per range so the skip-count stays scoped to this range.",
                    },
                ),
                "start_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000000,
                        "tooltip": "Index (into the sorted folder) of the first image to process. Lets you run a middle slice, e.g. start_index=30000 to skip the first 30k.",
                    },
                ),
                "count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000000,
                        "tooltip": "How many images (from start_index) to process. 0 = from start_index to the end of the folder.",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "Max images processed per queue. Bounds memory so huge folders (e.g. 70k FFHQ) won't OOM. Re-queue (or set the frontend Batch count high) to chew through the rest; each run auto-skips finished ones and takes the next batch.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load"
    CATEGORY = "image"
    DESCRIPTION = "Resumable, batched folder loader: skips inputs whose output already exists and returns only the next batch_size worth."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The output depends on filesystem state (which output files already
        # exist), which is NOT captured by ComfyUI's widget-value cache key.
        # Returning NaN forces re-execution on every queue so the folder is
        # re-read and the skip logic actually runs. Without this, re-queuing an
        # identical prompt is served from cache and never re-reads the disk.
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
                f"Input folder not found: {where!r} "
                f"(resolved to {in_dir}). If this is a later stage, run the "
                f"previous stage first so its output is produced here."
            )
        all_files = sorted(
            f for f in os.listdir(in_dir) if f.lower().endswith(_VALID_EXT)
        )
        if not all_files:
            raise ValueError(f"No images found in input folder: {input_folder!r}")
        # Select the working range [start_index, start_index + count).
        # count <= 0 means "from start_index to the end of the folder".
        end = len(all_files) if count <= 0 else start_index + count
        inputs = all_files[start_index:end]
        if not inputs:
            raise ValueError(
                f"Range is empty: start_index={start_index}, count={count}, "
                f"but the folder only has {len(all_files)} images."
            )

        out_root = folder_paths.get_output_directory()
        sub, name = _split_prefix(done_prefix)
        done_dir = _safe_subdir(out_root, sub) if sub else os.path.realpath(out_root)

        done = 0
        if os.path.isdir(done_dir):
            done = sum(
                1
                for f in os.listdir(done_dir)
                if f.startswith(name + "_") and f.lower().endswith(".png")
            )

        remaining = inputs[done : done + batch_size]
        logging.info(
            "[ResumeLoader] folder=%r total=%d range=[%d:%d] working=%d done=%d batch=%d -> %d",
            input_folder, len(all_files), start_index, end, len(inputs), done, batch_size, len(remaining),
        )
        print(
            f"[ResumeLoader] {input_folder!r}: folder={len(all_files)}, "
            f"range=[{start_index}:{end}] working={len(inputs)}, "
            f"done={done}, batch={batch_size} -> processing {len(remaining)} this run"
        )

        if not remaining:
            # Everything is already done. Returning an empty list makes
            # downstream nodes like SaveImage IndexError on the empty batch
            # (ComfyUI's list iteration does images[0]). Return a silent
            # ExecutionBlocker instead: the whole downstream chain is skipped
            # with no error and no garbage output, so re-queuing a finished
            # stage (or a frontend Batch count that overshoots) is a clean
            # no-op rather than a stream of red errors.
            from comfy_execution.graph import ExecutionBlocker
            return (ExecutionBlocker(None), ExecutionBlocker(None))

        images = []
        masks = []
        for f in remaining:
            img = Image.open(os.path.join(in_dir, f))
            img = img.convert("RGBA")
            arr = np.array(img).astype(np.float32) / 255.0
            images.append(torch.from_numpy(arr[:, :, :3])[None,])
            masks.append(torch.from_numpy(arr[:, :, 3])[None,])

        return (images, masks)


NODE_CLASS_MAPPINGS = {
    "LoadUnprocessedImagesFromFolder": LoadUnprocessedImagesFromFolder,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
