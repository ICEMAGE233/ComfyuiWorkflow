"""Copy Part20-Part40 (20000-40000) A/C to GoatGroup/StickerProject.
One upload_folder call per Part (origin + illustration). Skips files already
present (upload_folder dedups). Stops if the 1000 commits/day limit is hit.

  python _copy_to_goat.py
"""
import os, shutil

COMFY = "C:/l00912806/ComfyUI/ComfyUI-aki-v3/ComfyUI-aki-v3/ComfyUI"
REPO = "GoatGroup/StickerProject"
TOKEN = "ms-7882d82a-c450-440d-b40f-00fe8cd3d8ba"
EP = "https://www.modelscope.ai"
CUTOUT = os.path.join(COMFY, "output", "cutout")
CDIR = os.path.join(COMFY, "output", "C")
START, END = 20000, 40000  # Parts 20-40

os.environ["MODELSCOPE_API_TOKEN"] = TOKEN
os.environ["MODELSCOPE_ENDPOINT"] = EP
from modelscope_hub.api import HubApi
api = HubApi(); api._config.endpoint = EP; api._config.token = TOKEN

def part_indices(nn):
    lo = max(nn * 1000, START); hi = min(nn * 1000 + 999, END)
    return range(lo, hi + 1)

def upload_part(nn):
    for folder, src in (("origin", CUTOUT), ("illustration", CDIR)):
        staging = os.path.join(COMFY, "output", f"_stage_goat_{folder}")
        if os.path.isdir(staging): shutil.rmtree(staging)
        part_dir = os.path.join(staging, f"Part{nn:02d}")
        os.makedirs(part_dir, exist_ok=True)
        n = 0
        for idx in part_indices(nn):
            s = os.path.join(src, f"{idx:05d}.png")
            if os.path.exists(s):
                try: os.link(s, os.path.join(part_dir, f"{idx:05d}.png"))
                except OSError: shutil.copy(s, os.path.join(part_dir, f"{idx:05d}.png"))
                n += 1
        print(f"[Part{nn:02d}] {folder}: staging {n} files -> GoatGroup", flush=True)
        r = api.upload_folder(REPO, "dataset", staging, path_in_repo=folder,
                              commit_message=f"Part{nn:02d} {folder} (copy)")
        print(f"[Part{nn:02d}] {folder} done: {str(r)[:80]}", flush=True)
        shutil.rmtree(staging)

for nn in range(START // 1000, END // 1000 + 1):  # 20..40
    try:
        upload_part(nn)
        print(f"[Part{nn:02d}] COMPLETE", flush=True)
    except Exception as e:
        msg = repr(e)
        print(f"[Part{nn:02d}] FAILED: {msg[:250]}", flush=True)
        if "429" in msg or "limit" in msg.lower() or "frequency" in msg.lower():
            print("!! commit limit hit on GoatGroup, stopping", flush=True)
            break
print("DONE copy to GoatGroup", flush=True)
