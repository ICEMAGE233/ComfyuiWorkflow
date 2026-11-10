"""3D-anime-style generation: reuse cutout/ (A), FLUX with the 3D-anime prompt
(from anime_style_seg node 32), VAE = full_encoder_small_decoder. Saves B_anime + C_anime.

  python _runrange_anime.py <start> <end> [batch=10]
"""
import os, sys, json, time, uuid, urllib.request

COMFY = "C:/l00912806/ComfyUI/ComfyUI-aki-v3/ComfyUI-aki-v3/ComfyUI"
BASE = "http://127.0.0.1:8188"
WF = os.path.join(COMFY, "user", "default", "workflows", "flux2-V3-保存分割-仅插画.json")
ANIME_PROMPT = open("C:/Users/Persona/_anime_prompt.txt", encoding="utf-8").read().strip()

START = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
END = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 10

wf = json.load(open(WF, encoding="utf-8"))
subdefs = {s['id']: s for s in wf['definitions']['subgraphs']}
top_links = {l[0]: l for l in wf['links']}
by = {n["id"]: n for n in wf['nodes']}

# override the FLUX subgraph instance (403): 3D-anime prompt + full_encoder_small_decoder VAE
by[403]["widgets_values"][2] = "full_encoder_small_decoder.safetensors"  # vae_name
by[403]["widgets_values"][3] = ANIME_PROMPT                              # text
by[403]["widgets_values_named"]["vae_name"] = "full_encoder_small_decoder.safetensors"
by[403]["widgets_values_named"]["text"] = ANIME_PROMPT

api = {}; output_map = {}
uidc = [9200]
def newid(b): uidc[0] += 1; return f"{b}_{uidc[0]}"
def resolve_top(link_id):
    l = top_links[link_id]; src, slot = l[1], l[2]
    return output_map.get((src, slot), [str(src), slot])
def build_inputs(node, resolver):
    d = {}; wv = list(node.get('widgets_values', [])); wi = 0
    for inp in node.get('inputs', []):
        key = (inp.get('widget') or {}).get('name') or inp['name']
        link = inp.get('link'); is_w = 'widget' in inp
        if link is not None: d[key] = resolver(link)
        elif is_w: d[key] = wv[wi] if wi < len(wv) else None
        if is_w: wi += 1
    return d
def expand_instance(inst):
    sd = subdefs[inst['type']]; iid = inst['id']
    ext = {}; wv = list(inst.get('widgets_values', [])); wi = 0
    for i, inp in enumerate(inst.get('inputs', [])):
        link = inp.get('link'); is_w = 'widget' in inp
        if link is not None: ext[i] = resolve_top(link)
        elif is_w: ext[i] = wv[wi] if wi < len(wv) else None
        if is_w: wi += 1
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

def build_prompt(start, count):
    api.clear(); output_map.clear(); uidc[0] = 9200
    # re-apply override (by[403] is shared, re-apply each build)
    by[403]["widgets_values"][2] = "full_encoder_small_decoder.safetensors"
    by[403]["widgets_values"][3] = ANIME_PROMPT
    by[403]["widgets_values_named"]["vae_name"] = "full_encoder_small_decoder.safetensors"
    by[403]["widgets_values_named"]["text"] = ANIME_PROMPT
    # 1. loader (reuse cutout)
    api["521"] = {"class_type": "LoadUnprocessedImagesFromFolder",
                  "inputs": {"read_from": "output", "input_folder": "cutout", "done_prefix": "C_anime",
                             "start_index": start, "count": count, "batch_size": count}}
    # 2. EmptyImage (black canvas)
    api["390"] = {"class_type": by[390]['type'], "inputs": build_inputs(by[390], resolve_top)}
    # 3. composite (rewired: dest=390, source/mask from loader — skip 388 BiRefNet A)
    api["389"] = {"class_type": "ImageCompositeMasked",
                  "inputs": {"destination": ["390", 0], "source": ["521", 0], "mask": ["521", 1],
                             "x": 0, "y": 0, "resize_source": False}}
    # 4. FLUX (expand, with 3D-anime prompt + full_encoder_small_decoder VAE)
    expand_instance(by[403])
    # 5. scale
    api["518"] = {"class_type": by[518]['type'], "inputs": build_inputs(by[518], resolve_top)}
    # 6. saturation
    api["517"] = {"class_type": by[517]['type'], "inputs": build_inputs(by[517], resolve_top)}
    # 7. SaveImageIndexed B_anime
    api["401"] = {"class_type": "SaveImageIndexed", "inputs": {"image": ["517", 0], "index": ["521", 2], "prefix": "B_anime"}}
    # 8. BiRefNet C (expand, takes B from 401)
    expand_instance(by[545])
    # 9. SaveImageIndexed C_anime
    C_out = output_map[(545, 0)]
    api["546"] = {"class_type": "SaveImageIndexed", "inputs": {"image": C_out, "index": ["521", 2], "prefix": "C_anime"}}
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
print("DONE anime generation", flush=True)
