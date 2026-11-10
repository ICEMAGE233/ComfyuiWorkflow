"""Watch local C_pixel/ folder; upload each completed Part to ICEMAGE/StickerProject
under pixel/PartNN/ via upload_folder (batched commits). Runs alongside _runrange_pixel.py.

  python _partwatcher_pixel.py <start> <end>
e.g. python _partwatcher_pixel.py 20000 40000
"""
import os, sys, time, shutil

COMFY = "C:/l00912806/ComfyUI/ComfyUI-aki-v3/ComfyUI-aki-v3/ComfyUI"
REPO = "ICEMAGE/StickerProject"
TOKEN = "ms-7882d82a-c450-440d-b40f-00fe8cd3d8ba"
EP = "https://www.modelscope.ai"
START = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
END = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
MARKER = os.path.join(COMFY, "output", "_uploaded_parts_pixel.txt")
CDIR = os.path.join(COMFY, "output", "C_pixel")
FOLDER = "pixel"

os.environ["MODELSCOPE_API_TOKEN"] = TOKEN
os.environ["MODELSCOPE_ENDPOINT"] = EP
from modelscope_hub.api import HubApi
api = HubApi(); api._config.endpoint = EP; api._config.token = TOKEN

def uploaded_parts():
    if not os.path.exists(MARKER): return set()
    return set(int(l.strip()) for l in open(MARKER) if l.strip().isdigit())

def part_indices(nn):
    lo = max(nn * 1000, START); hi = min(nn * 1000 + 999, END)
    return range(lo, hi + 1)

def part_complete(nn):
    for idx in part_indices(nn):
        if not os.path.exists(os.path.join(CDIR, f"{idx:05d}.png")): return False
    return True

def upload_part(nn):
    staging = os.path.join(COMFY, "output", f"_stage_pix_{nn}")
    if os.path.isdir(staging): shutil.rmtree(staging)
    part_dir = os.path.join(staging, f"Part{nn:02d}"); os.makedirs(part_dir, exist_ok=True)
    n = 0
    for idx in part_indices(nn):
        s = os.path.join(CDIR, f"{idx:05d}.png")
        if os.path.exists(s):
            try: os.link(s, os.path.join(part_dir, f"{idx:05d}.png"))
            except OSError: shutil.copy(s, os.path.join(part_dir, f"{idx:05d}.png"))
            n += 1
    print(f"[Part{nn:02d}] uploading {n} files -> {FOLDER}/Part{nn:02d} ...", flush=True)
    r = api.upload_folder(REPO, "dataset", staging, path_in_repo=FOLDER, commit_message=f"Part{nn:02d} {FOLDER}")
    print(f"[Part{nn:02d}] done: {str(r)[:80]}", flush=True)
    shutil.rmtree(staging)
    with open(MARKER, "a") as f: f.write(f"{nn}\n")

all_parts = set(range(START // 1000, END // 1000 + 1))
done = uploaded_parts()
print(f"watching Parts {sorted(all_parts)} (already uploaded: {sorted(done)})", flush=True)
print(f"range {START}-{END}, folder={FOLDER}/, repo={REPO}, checking every 60s", flush=True)

while True:
    progressed = False
    for nn in sorted(all_parts):
        if nn in done: continue
        if part_complete(nn):
            try:
                upload_part(nn); done.add(nn); progressed = True
                print(f"[Part{nn:02d}] uploaded. done: {sorted(done)} / {sorted(all_parts)}", flush=True)
            except Exception as e:
                print(f"[Part{nn:02d}] FAILED: {repr(e)[:200]}", flush=True)
    if all_parts <= done:
        print("all Parts uploaded, exiting", flush=True); break
    if not progressed:
        try:
            files = [f for f in os.listdir(CDIR) if f.endswith('.png')]
            latest = max(int(f[:-4]) for f in files) if files else 0
            print(f"  ...waiting, C_pixel at idx {latest}", flush=True)
        except Exception: pass
    time.sleep(60)
