"""Re-score ONLY relation accuracy on the saved RQ5 images, using the fixed
parse_spatial_relations (object = head noun, not adjective). Writes per-image
results to outputs/sdxl/rq5/relation_rescore.jsonl for significance testing."""
import json, re, statistics as st
from pathlib import Path
from PIL import Image
from evaluation.metrics import RelationAccuracyScorer

OUT = Path("outputs/sdxl/rq5")
prompts = {}
for l in open("outputs/sdxl/rq5_text/text_pairs.jsonl"):
    r = json.loads(l); prompts[r["idx"]] = r["prompt"]

scorer = RelationAccuracyScorer()
conds = ["raw", "ar", "llada_single", "llada_poe"]
res = {c: {} for c in conds}
recs = []
for c in conds:
    pngs = sorted(p for p in (OUT / c).glob("prompt_*.png"))
    print(f"[{c}] {len(pngs)} images", flush=True)
    for png in pngs:
        m = re.match(r"prompt_(\d+)_", png.name)
        if not m:
            continue
        idx = int(m.group(1))
        pr = prompts.get(idx)
        if pr is None:
            continue
        d = scorer.score(Image.open(png).convert("RGB"), pr)
        acc = d["accuracy"] if d["n_relations"] > 0 else None
        res[c][idx] = acc
        recs.append({"cond": c, "idx": idx, "rel_acc": acc, "n_rel": d["n_relations"]})

with open(OUT / "relation_rescore.jsonl", "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")

print("=== RELATION ACCURACY (re-scored, fixed parser) ===")
for c in conds:
    vals = [v for v in res[c].values() if v is not None]
    print(f"{c:14s} rel_acc={st.mean(vals):.3f} (n={len(vals)})")
def paired(a, b):
    idxs = [i for i in res[a] if i in res[b] and res[a][i] is not None and res[b][i] is not None]
    d = [res[a][i] - res[b][i] for i in idxs]
    w = sum(x > 1e-9 for x in d); l = sum(x < -1e-9 for x in d); t = sum(abs(x) <= 1e-9 for x in d)
    print(f"  {a} vs {b:13s}: delta={st.mean(d):+.3f}  W/L/T={w}/{l}/{t}  (n={len(idxs)})")
for b in ["raw", "ar", "llada_single"]:
    paired("llada_poe", b)
print("DONE_RESCORE")
