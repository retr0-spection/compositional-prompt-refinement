"""RQ5 x RQ6 -- pipeline CLIP robustness vs guidance scale.

Final-image CLIPScore across a CFG grid (incl. an overshoot tail to 50), per
conditioning pipeline (raw / single / PoE). Tests whether diffusion refinement
yields a flatter, more robust CLIP-vs-CFG curve: if refinement lowers the
prompt's "error signal", refined conditioning should SUSTAIN alignment as
guidance is pushed past saturation, whereas raw peaks early and collapses.

Distinct from experiments/cfg_stability.py (per-step trajectory smoothness) and
from rq3_cfg_sensitivity (variance of final quality). This one = endpoint CLIP
vs CFG, compared across pipelines.

Output: outputs/sdxl/rq5_cfg_robustness/clip_vs_cfg.jsonl  {cond, cfg, idx, clip}
Resumable: appends + skips (cond, cfg, idx) already written.
"""
import json
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline
from evaluation.metrics import CLIPScorer

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
STEPS = 50
CFG_GRID = [3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]  # 20+ = overshoot
SEED = 42
N_PROMPTS = 15
OUT = Path("outputs/sdxl/rq5_cfg_robustness")
OUT.mkdir(parents=True, exist_ok=True)

prompts, single, poe = [], {}, {}
for line in open("outputs/sdxl/rq5_text/text_pairs.jsonl"):
    r = json.loads(line)
    prompts.append(r["prompt"]); single[r["idx"]] = r["single_text"]; poe[r["idx"]] = r["poe_text"]
idxs = list(range(min(N_PROMPTS, len(prompts))))

print("loading SDXL...", flush=True)
pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True)
pipe.upcast_vae()  # SDXL fp16 VAE stability for the final decode
clip = CLIPScorer()

conditions = {"raw": lambda i: prompts[i], "single": lambda i: single[i], "poe": lambda i: poe[i]}

outfile = OUT / "clip_vs_cfg.jsonl"
done = set()
if outfile.exists():
    for line in open(outfile):
        try:
            r = json.loads(line); done.add((r["cond"], float(r["cfg"]), int(r["idx"])))
        except Exception:
            pass

f = open(outfile, "a")
for cfg in CFG_GRID:
    for cond, textfn in conditions.items():
        print("[cfg=" + str(cfg) + " " + cond + "]", flush=True)
        for i in idxs:
            if (cond, float(cfg), i) in done:
                continue
            g = torch.Generator("cuda").manual_seed(SEED + i)
            out = pipe(prompt=textfn(i), num_inference_steps=STEPS, guidance_scale=cfg, generator=g)
            im = out.images[0]
            rec = {"cond": cond, "cfg": float(cfg), "idx": i,
                   "clip": float(clip.score(im, prompts[i]))}
            f.write(json.dumps(rec) + "\n"); f.flush()
f.close()
print("DONE_RQ5_CFG_ROBUSTNESS", flush=True)
