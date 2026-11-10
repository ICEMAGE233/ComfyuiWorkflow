"""Local-only generation over an FFHQ range (NO upload). Saves cutout/B/C locally.
Use after the per-file upload hit the ModelScope 1000 commits/day limit.
Resume-safe (skips inputs whose cutout/{idx}.png already exists).

  python _runrange_local.py <start> <end> [batch=10]
e.g. python _runrange_local.py 31540 40000 10
"""
import os, sys, json, time, uuid, urllib.request

COMFY = "C:/l00912806/ComfyUI/ComfyUI-aki-v3/ComfyUI-aki-v3/ComfyUI"
BASE = "http://127.0.0.1:8188"
WF = os.path.join(COMFY, "user", "default", "workflows", "flux2-V3-保存分割-仅插画.json")

START = int(sys.argv[1]) if len(sys.argv) > 1 else 31540
END = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 10

wf = json.load(open(WF, encoding="utf-8"))
subdefs = {s['id']: s for s in wf['definitions']['subgraphs']}
top_links = {l[0]: l for l in wf['links']}
active = set()
for l in wf['links']:
    active.add(l[1]); active.add(l[3])

def build_prompt(start, count):
    api = {}; output_map = {}
    uidc = [9000]
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
    for n in sorted(wf['nodes'], key=lambda x: x.get('order', 0)):
        nid = n['id']
        if nid in (521, 391, 401, 546, 396) or nid not in active: continue
        if n['type'] in subdefs: expand_instance(n)
        else: api[str(nid)] = {"class_type": n['type'], "inputs": build_inputs(n, resolve_top)}
    A = output_map[(388, 0)]; B = ["517", 0]; C = output_map[(545, 0)]
    api["521"] = {"class_type": "LoadUnprocessedImagesFromFolder",
                  "inputs": {"read_from": "input", "input_folder": "ffhq/images1024x1024", "done_prefix": "cutout",
                             "start_index": start, "count": count, "batch_size": count}}
    api["391"] = {"class_type": "SaveImageIndexed", "inputs": {"image": A, "index": ["521", 2], "prefix": "cutout"}}
    api["401"] = {"class_type": "SaveImageIndexed", "inputs": {"image": B, "index": ["521", 2], "prefix": "B"}}
    api["546"] = {"class_type": "SaveImageIndexed", "inputs": {"image": C, "index": ["521", 2], "prefix": "C"}}
    return api  # NO UploadToModelScope nodes -> no per-file commits, no 429

def post_wait(api_prompt, timeout=600):
    body = json.dumps({"prompt": api_prompt, "client_id": uuid.uuid4().hex}).encode()
    req = urllib.request.Request(BASE+"/prompt", data=body, headers={"Content-Type":"application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=20); res = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return "HTTP"+str(e.code), e.read().decode("utf-8","replace")[:300]
    if res.get("node_errors") or "error" in res:
        return "REJECTED", json.dumps(res, ensure_ascii=False)[:300]
    pid = res['prompt_id']; t0 = time.time()
    while time.time()-t0 < timeout:
        try: h = json.loads(urllib.request.urlopen(BASE+f"/history/{pid}", timeout=10).read())
        except: h = {}
        if pid in h:
            d = h[pid]
            if any(m[0]=='execution_error' for m in d.get('status',{}).get('messages',[])):
                return "EXEC_ERROR", ""
            return "ok", None
        time.sleep(1.0)
    return "TIMEOUT", None

total_batches = (END - START + BATCH) // BATCH
b = 0; t_start = time.time()
try:
    for start in range(START, END + 1, BATCH):
        count = min(BATCH, END - start + 1)
        b += 1
        st, info = post_wait(build_prompt(start, count))
        elapsed = time.time() - t_start
        eta = elapsed / b * (total_batches - b) if b else 0
        print(f"[{b}/{total_batches}] idx {start}-{start+count-1}  {st}  elapsed={elapsed:.0f}s eta={eta/3600:.1f}h" + (f" info={info}" if st!='ok' else ''))
        if st in ('REJECTED','EXEC_ERROR','HTTP400','HTTP500'):
            print("  fatal, stopping"); break
except KeyboardInterrupt:
    print("\n interrupted, local files are safe")
print("DONE local generation")
