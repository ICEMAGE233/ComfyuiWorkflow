"""Copy inflated Part20-40 (C_inflated) to GoatGroup/StickerProject under inflated/.
One upload_folder per Part. LFS blobs dedup globally (already on ICEMAGE) -> fast.

  python _copy_inflated_to_goat.py
"""
import os, shutil

COMFY = "C:/l00912806/ComfyUI/ComfyUI-aki-v3/ComfyUI-aki-v3/ComfyUI"
REPO = "GoatGroup/StickerProject"
TOKEN = "ms-7882d82a-c450-440d-b40f-00fe8cd3d8ba"
EP = "https://www.modelscope.ai"
SRC = os.path.join(COMFY, "output", "C_inflated")
FOLDER = "inflated"
START, END = 20000, 40000

os.environ["MODELSCOPE_API_TOKEN"] = TOKEN
os.environ["MODELSCOPE_ENDPOINT"] = EP
from modelscope_hub.api import HubApi
api = HubApi(); api._config.endpoint = EP; api._config.token = TOKEN

def part_indices(nn):
    lo = max(nn * 1000, START); hi = min(nn * 1000 + 999, END)
    return range(lo, hi + 1)

for nn in range(START // 1000, END // 1000 + 1):
    staging = os.path.join(COMFY, "output", f"_stage_goat_inf")
    if os.path.isdir(staging): shutil.rmtree(staging)
    part_dir = os.path.join(staging, f"Part{nn:02d}"); os.makedirs(part_dir, exist_ok=True)
    n = 0
    for idx in part_indices(nn):
        s = os.path.join(SRC, f"{idx:05d}.png")
        if os.path.exists(s):
            try: os.link(s, os.path.join(part_dir, f"{idx:05d}.png"))
            except OSError: shutil.copy(s, os.path.join(part_dir, f"{idx:05d}.png"))
            n += 1
    print(f"[Part{nn:02d}] {n} files -> GoatGroup {FOLDER}/", flush=True)
    try:
        r = api.upload_folder(REPO, "dataset", staging, path_in_repo=FOLDER, commit_message=f"Part{nn:02d} {FOLDER} (copy)")
        print(f"[Part{nn:02d}] done: {str(r)[:80]}", flush=True)
    except Exception as e:
        print(f"[Part{nn:02d}] FAILED: {repr(e)[:200]}", flush=True)
        if "429" in str(e) or "limit" in str(e).lower(): print("limit hit, stopping", flush=True); break
    shutil.rmtree(staging)
print("DONE copy inflated to GoatGroup", flush=True)
