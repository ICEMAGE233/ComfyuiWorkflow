"""Pixel-style generation: reuse cutout/ (A), pixelate via PixelArtDetectorToImage,
save C_pixel/{idx}.png locally (NO upload). Resume-safe (done_prefix=C_pixel).

Chain: cutout -> ImageScaleBy(0.35 nearest) -> PixelArtDetectorToImage(downscale_to_1x)
       -> ImageScale(2048 nearest) -> SaveImageIndexed(C_pixel)

  python _runrange_pixel.py <start> <end> [batch=10]
e.g. python _runrange_pixel.py 20000 40000 10
"""
import os, sys, json, time, uuid, urllib.request

COMFY = "C:/l00912806/ComfyUI/ComfyUI-aki-v3/ComfyUI-aki-v3/ComfyUI"
BASE = "http://127.0.0.1:8188"
START = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
END = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 10

def build_prompt(start, count):
    return {
        "load": {"class_type": "LoadUnprocessedImagesFromFolder",
                 "inputs": {"read_from": "output", "input_folder": "cutout", "done_prefix": "C_pixel",
                            "start_index": start, "count": count, "batch_size": count}},
        "scale": {"class_type": "ImageScaleBy",
                  "inputs": {"image": ["load", 0], "upscale_method": "nearest-exact", "scale_by": 0.35}},
        "pix": {"class_type": "PixelArtDetectorToImage",
                "inputs": {"images": ["scale", 0], "reduce_palette": False, "reduce_palette_max_colors": 64}},
        "resize": {"class_type": "ImageScale",
                   "inputs": {"image": ["pix", 0], "upscale_method": "nearest-exact", "width": 2048, "height": 2048, "crop": "disabled"}},
        "save": {"class_type": "SaveImageIndexed",
                 "inputs": {"image": ["resize", 0], "index": ["load", 2], "prefix": "C_pixel"}},
    }

def post_wait(prompt, timeout=600):
    body = json.dumps({"prompt": prompt, "client_id": uuid.uuid4().hex}).encode()
    req = urllib.request.Request(BASE+"/prompt", data=body, headers={"Content-Type":"application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=20); res = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return "HTTP"+str(e.code), e.read().decode("utf-8","replace")[:300]
    if res.get("node_errors") or "error" in res:
        return "REJECTED", json.dumps(res, ensure_ascii=False)[:400]
    pid = res['prompt_id']; t0 = time.time()
    while time.time()-t0 < timeout:
        try: h = json.loads(urllib.request.urlopen(BASE+f"/history/{pid}", timeout=10).read())
        except: h = {}
        if pid in h:
            d = h[pid]
            if any(m[0]=='execution_error' for m in d.get('status',{}).get('messages',[])):
                errs=[m for m in d['status']['messages'] if m[0]=='execution_error']
                return "EXEC_ERROR", json.dumps(errs, ensure_ascii=False)[:400]
            return "ok", None
        time.sleep(1.0)
    return "TIMEOUT", None

total = (END - START + BATCH) // BATCH
b = 0; t0 = time.time()
try:
    for start in range(START, END + 1, BATCH):
        count = min(BATCH, END - start + 1); b += 1
        st, info = post_wait(build_prompt(start, count))
        el = time.time() - t0; eta = el / b * (total - b) if b else 0
        print(f"[{b}/{total}] idx {start}-{start+count-1}  {st}  elapsed={el:.0f}s eta={eta/3600:.1f}h" + (f" {info}" if st!='ok' else ''), flush=True)
        if st in ('REJECTED','EXEC_ERROR','HTTP400','HTTP500'):
            print("  fatal, stopping"); break
except KeyboardInterrupt:
    print("\n interrupted, local files safe", flush=True)
print("DONE pixel generation", flush=True)
