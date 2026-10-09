"""CFG stability analysis: final CLIPScore vs guidance scale, per pipeline.

Proposal metric ("CLIPScore stability with increasing CFG across pipelines").
Sweep CFG over a grid; for each (condition, cfg) generate the FINAL SDXL image
for a prompt sample and score CLIPScore against the ORIGINAL prompt.

Hypothesis: if diffusion refinement (single / PoE) lowers the prompt's "error
signal" (constraints less ambiguous / less in competition), refined conditioning
should SUSTAIN alignment as guidance increases -- a flatter, more stable
CLIP-vs-CFG curve with a later/softer saturation knee -- whereas the raw prompt
peaks early and collapses into oversaturation at high CFG.

Output: outputs/sdxl/diagnostics/cfg_stability.jsonl  {cond, cfg, idx, clip}
Resumable: appends + skips (cond, cfg, idx) already written.
"""
import json
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline
from evaluation.metrics import CLIPScorer

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
STEPS = 50
CFG_GRID = [3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]
SEED = 42
N_PROMPTS = 15
OUT = Path("outputs/sdxl/diagnostics")
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

outfile = OUT / "cfg_stability.jsonl"
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
print("DONE_STABILITY", flush=True)
