"""Watch local cutout/C folders; upload each Part (1000 images) to ICEMAGE via
upload_folder as soon as it's complete. Runs alongside _runrange_local.py.

  python _partwatcher.py <start> <end>
e.g. python _partwatcher.py 31540 40000

For Part NN, the indices are [NN*1000, NN*1000+999] intersected with [start,end].
A Part is "complete" when all its indices exist in BOTH cutout/ and C/.
On completion, stages those files and upload_folder them to origin/PartNN and
illustration/PartNN (batched commit, ~4 commits per Part -> well under the
1000/day limit). Tracks done Parts in output/_uploaded_parts.txt.
"""
import os, sys, time, shutil

COMFY = "C:/l00912806/ComfyUI/ComfyUI-aki-v3/ComfyUI-aki-v3/ComfyUI"
REPO = "ICEMAGE/StickerProject"
TOKEN = "ms-7882d82a-c450-440d-b40f-00fe8cd3d8ba"
EP = "https://www.modelscope.ai"
START = int(sys.argv[1]) if len(sys.argv) > 1 else 31540
END = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
MARKER = os.path.join(COMFY, "output", "_uploaded_parts.txt")
CUTOUT = os.path.join(COMFY, "output", "cutout")
CDIR = os.path.join(COMFY, "output", "C")

os.environ["MODELSCOPE_API_TOKEN"] = TOKEN
os.environ["MODELSCOPE_ENDPOINT"] = EP
from modelscope_hub.api import HubApi
api = HubApi(); api._config.endpoint = EP; api._config.token = TOKEN

def uploaded_parts():
    if not os.path.exists(MARKER):
        return set()
    return set(int(l.strip()) for l in open(MARKER) if l.strip().isdigit())

def part_indices(nn):
    lo = max(nn * 1000, START)
    hi = min(nn * 1000 + 999, END)
    return range(lo, hi + 1)

def part_complete(nn):
    for idx in part_indices(nn):
        if not os.path.exists(os.path.join(CUTOUT, f"{idx:05d}.png")):
            return False
        if not os.path.exists(os.path.join(CDIR, f"{idx:05d}.png")):
            return False
    return True

def upload_part(nn):
    # stage this Part's cutout + C, upload_folder each (batched commits)
    for folder, src in (("origin", CUTOUT), ("illustration", CDIR)):
        staging = os.path.join(COMFY, "output", f"_stage_{folder}")
        if os.path.isdir(staging):
            shutil.rmtree(staging)
        part_dir = os.path.join(staging, f"Part{nn:02d}")
        os.makedirs(part_dir, exist_ok=True)
        n = 0
        for idx in part_indices(nn):
            s = os.path.join(src, f"{idx:05d}.png")
            if os.path.exists(s):
                try:
                    os.link(s, os.path.join(part_dir, f"{idx:05d}.png"))
                except OSError:
                    shutil.copy(s, os.path.join(part_dir, f"{idx:05d}.png"))
                n += 1
        print(f"[Part{nn:02d}] uploading {n} files -> {folder}/Part{nn:02d} ...", flush=True)
        r = api.upload_folder(REPO, "dataset", staging, path_in_repo=folder,
                              commit_message=f"Part{nn:02d} {folder}")
        print(f"[Part{nn:02d}] {folder} done: {str(r)[:80]}", flush=True)
        shutil.rmtree(staging)
    with open(MARKER, "a") as f:
        f.write(f"{nn}\n")

all_parts = set(range(START // 1000, END // 1000 + 1))
done = uploaded_parts()
print(f"watching Parts {sorted(all_parts)} (already uploaded: {sorted(done)})", flush=True)
print(f"range {START}-{END}, checking every 60s", flush=True)

while True:
    progressed = False
    for nn in sorted(all_parts):
        if nn in done:
            continue
        if part_complete(nn):
            try:
                upload_part(nn)
                done.add(nn)
                progressed = True
                print(f"[Part{nn:02d}] uploaded. done: {sorted(done)} / {sorted(all_parts)}", flush=True)
            except Exception as e:
                print(f"[Part{nn:02d}] upload FAILED: {repr(e)[:200]}", flush=True)
    if all_parts <= done:
        print("all Parts uploaded, exiting", flush=True)
        break
    if not progressed:
        # brief status: how far cutout has reached
        try:
            files = sorted(os.listdir(CUTOUT))
            latest = max(int(f[:-4]) for f in files if f.endswith(".png")) if files else 0
            print(f"  ...waiting, cutout at idx {latest} (next Part completes at {((latest//1000)+1)*1000})", flush=True)
        except Exception:
            pass
    time.sleep(60)
