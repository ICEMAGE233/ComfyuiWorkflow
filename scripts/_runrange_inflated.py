"""Inflated-style generation: reuse existing cutout/ (A), run FLUX with the
inflated-vinyl-balloon prompt, save B_inflated + C_inflated locally (NO upload).
Resume-safe (done_prefix=C_inflated, skips idx whose C_inflated/{idx}.png exists).

  python _runrange_inflated.py <start> <end> [batch=10]
e.g. python _runrange_inflated.py 20000 40000 10
"""
import os, sys, json, time, uuid, urllib.request

COMFY = "C:/l00912806/ComfyUI/ComfyUI-aki-v3/ComfyUI-aki-v3/ComfyUI"
BASE = "http://127.0.0.1:8188"
WF = os.path.join(COMFY, "user", "default", "workflows", "flux2-V3-保存分割-仅膨胀.json")

START = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
END = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 10

wf = json.load(open(WF, encoding="utf-8"))
subdefs = {s['id']: s for s in wf['definitions']['subgraphs']}
top_links = {l[0]: l for l in wf['links']}

api = {}; output_map = {}
uidc = [9100]
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

by = {n["id"]: n for n in wf['nodes']}

def build_prompt(start, count):
    api.clear(); output_map.clear(); uidc[0] = 9100
    # build in dependency order: subgraphs must be expanded before regular nodes that reference them
    # 1. loader (reuse cutout/ as A)
    api["558"] = {"class_type": "LoadUnprocessedImagesFromFolder",
                  "inputs": {"read_from": "output", "input_folder": "cutout", "done_prefix": "C_inflated",
                             "start_index": start, "count": count, "batch_size": count}}
    # 2. EmptyImage (black 1024x1024)
    api["550"] = {"class_type": by[550]['type'], "inputs": build_inputs(by[550], resolve_top)}
    # 3. composite (rewired: dest=550, source/mask from loader — reuse A, skip 557)
    api["551"] = {"class_type": "ImageCompositeMasked",
                  "inputs": {"destination": ["550", 0], "source": ["558", 0], "mask": ["558", 1],
                             "x": 0, "y": 0, "resize_source": False}}
    # 4. FLUX (expand) — depends on 551; carries the inflated prompt + 561 brightness downstream
    expand_instance(by[555])
    # 5. scale — depends on 555 (must be after expand so output_map[(555,0)] is set)
    api["554"] = {"class_type": by[554]['type'], "inputs": build_inputs(by[554], resolve_top)}
    # 6. brightness (1.1/1.05/1.25) — depends on 554
    api["561"] = {"class_type": by[561]['type'], "inputs": build_inputs(by[561], resolve_top)}
    # 7. SaveImageIndexed B_inflated — depends on 561
    api["552"] = {"class_type": "SaveImageIndexed", "inputs": {"image": ["561", 0], "index": ["558", 2], "prefix": "B_inflated"}}
    # 8. BiRefNet C (expand) — depends on 552 (pass-through B)
    expand_instance(by[553])
    # 9. SaveImageIndexed C_inflated — depends on 553
    C_out = output_map[(553, 0)]
    api["559"] = {"class_type": "SaveImageIndexed", "inputs": {"image": C_out, "index": ["558", 2], "prefix": "C_inflated"}}
    return dict(api)

def post_wait(api_prompt, timeout=600):
    body = json.dumps({"prompt": api_prompt, "client_id": uuid.uuid4().hex}).encode()
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
print("DONE inflated generation", flush=True)
