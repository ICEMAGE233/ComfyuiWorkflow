"""Batch-upload local cutout (A) and C to GoatGroup via upload_folder.

upload_folder packs many files into each commit (batch commits), so this uses
only a few dozen commits total instead of one-per-file -> bypasses the
ModelScope 1000 commits/day limit.

Run AFTER the commit limit has reset (the 429 said ~8.5h).

Stages:
  cutout/{idx}.png -> origin/Part{idx//1000:02d}/{idx:05d}.png
  C/{idx}.png      -> illustration/Part{idx//1000:02d}/{idx:05d}.png
(B is not uploaded.)
"""
import os, shutil

COMFY = "C:/l00912806/ComfyUI/ComfyUI-aki-v3/ComfyUI-aki-v3/ComfyUI"
REPO = "ICEMAGE/StickerProject"
TOKEN = "ms-7882d82a-c450-440d-b40f-00fe8cd3d8ba"
EP = "https://www.modelscope.ai"

os.environ["MODELSCOPE_API_TOKEN"] = TOKEN
os.environ["MODELSCOPE_ENDPOINT"] = EP
from modelscope_hub.api import HubApi
api = HubApi(); api._config.endpoint = EP; api._config.token = TOKEN

def stage(src_dir, folder):
    staging = os.path.join(COMFY, "output", f"_stage_{folder}")
    if os.path.isdir(staging): shutil.rmtree(staging)
    n = 0
    for fn in sorted(os.listdir(src_dir)):
        if not fn.endswith(".png"): continue
        idx = int(fn[:-4])
        part = f"Part{idx//1000:02d}"
        d = os.path.join(staging, part); os.makedirs(d, exist_ok=True)
        s = os.path.join(src_dir, fn); t = os.path.join(d, fn)
        try: os.link(s, t)           # hardlink: instant, no extra disk
        except OSError: shutil.copy(s, t)
        n += 1
    print(f"  staged {n} files -> {folder}/")
    return staging

for folder, src in (("origin", "cutout"), ("illustration", "C")):
    src_dir = os.path.join(COMFY, "output", src)
    if not os.path.isdir(src_dir):
        print(f"!! {src_dir} not found -> skip {folder}"); continue
    print(f"=== upload_folder -> {folder}/ ===")
    st = stage(src_dir, folder)
    r = api.upload_folder(REPO, "dataset", st, path_in_repo=folder, commit_message=f"batch upload {folder}")
    print(f"  {folder}: {r}")
    shutil.rmtree(st)
print("DONE batch upload")
