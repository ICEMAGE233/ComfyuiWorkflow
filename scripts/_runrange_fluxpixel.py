"""FLUX pixel-art generation: reuse cutout/ (A), FLUX(base-4b, pixel art prompt) 
→ ImageScaleBy(0.35) → PixelArtDetectorToImage → ImageScale(2048) → BiRefNet → C_pixel.
Resume-safe (done_prefix=C_pixel). NO upload (use watcher separately).

  python _runrange_fluxpixel.py <start> <end> [batch=10]
"""
import os, sys, json, time, uuid, urllib.request, copy

COMFY = "C:/l00912806/ComfyUI/ComfyUI-aki-v3/ComfyUI-aki-v3/ComfyUI"
BASE = "http://127.0.0.1:8188"
PIXEL_WF = os.path.join(COMFY, "user", "default", "workflows", "最终-像素风", "image_flux2_klein_image_edit_9b_base_4b.json")
ILLUST_WF = os.path.join(COMFY, "user", "default", "workflows", "flux2-V3-保存分割-仅插画.json")
PIXEL_PROMPT = open("C:/Users/Persona/_pixel_prompt.txt", encoding="utf-8").read().strip()

START = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
END = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 10

pwf = json.load(open(PIXEL_WF, encoding="utf-8"))
iwf = json.load(open(ILLUST_WF, encoding="utf-8"))
subdefs = {}
for s in pwf.get("definitions",{}).get("subgraphs",[]): subdefs[s['id']] = s
for s in iwf.get("definitions",{}).get("subgraphs",[]): subdefs[s['id']] = s
pby = {n["id"]: n for n in pwf["nodes"]}
iby = {n["id"]: n for n in iwf["nodes"]}
ptop_links = {l[0]: l for l in pwf["links"]}

api = {}; output_map = {}
uidc = [9500]
def newid(b): uidc[0] += 1; return f"{b}_{uidc[0]}"
def build_inputs(node, resolver):
    d = {}; wv = list(node.get('widgets_values', [])); wi = 0
    for inp in node.get('inputs', []):
        key = (inp.get('widget') or {}).get('name') or inp['name']
        link = inp.get('link'); is_w = 'widget' in inp
        if link is not None: d[key] = resolver(link)
        elif is_w: d[key] = wv[wi] if wi < len(wv) else None
        if is_w: wi += 1
    return d

def build_prompt(start, count):
    api.clear(); output_map.clear(); uidc[0] = 9500
    # 1. loader (reuse cutout)
    api["loader"] = {"class_type": "LoadUnprocessedImagesFromFolder",
                     "inputs": {"read_from": "output", "input_folder": "cutout", "done_prefix": "C_pixel",
                                "start_index": start, "count": count, "batch_size": count}}
    # 2. EmptyImage (black 1024x1024)
    api["empty"] = {"class_type": "EmptyImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1, "color": 0}}
    # 3. composite (dest=empty, source/mask from loader)
    api["composite"] = {"class_type": "ImageCompositeMasked",
                        "inputs": {"destination": ["empty", 0], "source": ["loader", 0], "mask": ["loader", 1],
                                   "x": 0, "y": 0, "resize_source": False}}
    # 4. FLUX (pixel art prompt, base-4b model, subgraph 65c22b29)
    flux_inst = copy.deepcopy(pby[92])
    flux_inst["widgets_values_named"]["text"] = PIXEL_PROMPT
    flux_inst["widgets_values_named"]["unet_name"] = "flux-2-klein-base-4b-fp8.safetensors"
    flux_inst["widgets_values_named"]["vae_name"] = "flux2-vae.safetensors"
    flux_inst["widgets_values_named"]["clip_name"] = "qwen_3_4b.safetensors"
    sd = subdefs[flux_inst['type']]
    ext = {}; wv = list(flux_inst.get('widgets_values', [])); wi = 0
    for i, inp in enumerate(flux_inst.get('inputs', [])):
        link = inp.get('link'); is_w = 'widget' in inp
        inp_name = inp.get('name', '')
        if inp_name in ('image', 'image_1'):
            ext[i] = ["composite", 0]
        elif link is not None:
            l = ptop_links[link]; src, slot = l[1], l[2]
            ext[i] = output_map.get((src, slot), [str(src), slot])
        elif is_w:
            wvn = flux_inst.get("widgets_values_named", {})
            key = (inp.get('widget') or {}).get('name', inp_name)
            ext[i] = wvn.get(key, wv[wi] if wi < len(wv) else None)
        if is_w: wi += 1
    iid = flux_inst['id']
    in_links = {l['id']: l for l in sd['links']}
    idmap = {n['id']: newid(f"sg{iid}") for n in sd['nodes']}
    in_id = sd['inputNode']['id']; out_id = sd['outputNode']['id']
    def ires(link_id):
        l = in_links[link_id]
        if l['origin_id'] == in_id: return ext[l['origin_slot']]
        return [idmap[l['origin_id']], l['origin_slot']]
    for n in sd['nodes']:
        api[idmap[n['id']]] = {"class_type": n['type'], "inputs": build_inputs(n, ires)}
    for l in sd['links']:
        if l['target_id'] == out_id:
            output_map[(iid, l['target_slot'])] = [idmap[l['origin_id']], l['origin_slot']]
    flux_out = output_map[(iid, 0)]
    # 5. PIXEL CHAIN: ImageScaleBy(0.35) → PixelArtDetectorToImage → ImageScale(2048)
    api["pix_scale"] = {"class_type": "ImageScaleBy",
                        "inputs": {"image": flux_out, "upscale_method": "nearest-exact", "scale_by": 0.35}}
    api["pix_detect"] = {"class_type": "PixelArtDetectorToImage",
                         "inputs": {"images": ["pix_scale", 0], "reduce_palette": False, "reduce_palette_max_colors": 64}}
    api["pix_resize"] = {"class_type": "ImageScale",
                         "inputs": {"image": ["pix_detect", 0], "upscale_method": "nearest-exact",
                                    "width": 2048, "height": 2048, "crop": "disabled"}}
    # 6. SaveImageIndexed B_pixel (FLUX pixel art output)
    api["saveB"] = {"class_type": "SaveImageIndexed",
                    "inputs": {"image": ["pix_resize", 0], "index": ["loader", 2], "prefix": "B_pixel"}}
    # 7. BiRefNet C (from illust workflow, subgraph 7171296b, node 545)
    biref_inst = iby[545]
    sd2 = subdefs[biref_inst['type']]
    ext2 = {}; wv2 = list(biref_inst.get('widgets_values', [])); wi2 = 0
    for i, inp in enumerate(biref_inst.get('inputs', [])):
        link = inp.get('link'); is_w = 'widget' in inp
        if inp.get('name') == 'image': ext2[i] = ["saveB", 0]
        elif link is not None:
            l = {l2[0]: l2 for l2 in iwf['links']}.get(link)
            if l: ext2[i] = output_map.get((l[1], l[2]), [str(l[1]), l[2]])
        elif is_w: ext2[i] = wv2[wi2] if wi2 < len(wv2) else None
        if is_w: wi2 += 1
    iid2 = biref_inst['id']
    in_links2 = {l['id']: l for l in sd2['links']}
    idmap2 = {n['id']: newid(f"sg{iid2}") for n in sd2['nodes']}
    in_id2 = sd2['inputNode']['id']; out_id2 = sd2['outputNode']['id']
    def ires2(link_id):
        l = in_links2[link_id]
        if l['origin_id'] == in_id2: return ext2[l['origin_slot']]
        return [idmap2[l['origin_id']], l['origin_slot']]
    for n in sd2['nodes']:
        api[idmap2[n['id']]] = {"class_type": n['type'], "inputs": build_inputs(n, ires2)}
    for l in sd2['links']:
        if l['target_id'] == out_id2:
            output_map[(iid2, l['target_slot'])] = [idmap2[l['origin_id']], l['origin_slot']]
    C_out = output_map[(iid2, 0)]
    # 8. SaveImageIndexed C_pixel
    api["saveC"] = {"class_type": "SaveImageIndexed",
                    "inputs": {"image": C_out, "index": ["loader", 2], "prefix": "C_pixel"}}
    return dict(api)

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
print("DONE flux-pixel generation", flush=True)
