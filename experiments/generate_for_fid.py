"""RQ5 x RQ6 -- generate SDXL images across a CFG grid for FID evaluation.

Saves PNGs to outputs/sdxl/fid/<pipeline>/cfg_<cfg>/<idx>_s<seed>.png.
Resumable: skips any image already written.
Multi-node sharding: if launched under `srun` with SLURM_NTASKS>1, each task
(one per Blackwell node) processes a disjoint slice of the work units, so the
two idle Blackwell GPUs run in parallel. Sharding is by flat work-unit index
modulo world size, which is deterministic and balanced.
No manual upcast_vae() -- the SDXL pipeline auto-upcasts its fp16 VAE.
"""
import os, json
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
STEPS = 50
CFGS  = [3.0, 7.5, 15.0, 30.0, 50.0]
SEEDS = [42, 43]
OUT   = Path("outputs/sdxl/fid")
CONDS = ["raw", "single", "poe"]

RANK  = int(os.environ.get("SLURM_PROCID", "0"))
WORLD = int(os.environ.get("SLURM_NTASKS", "1"))

prompts, single, poe = [], {}, {}
for line in open("outputs/sdxl/rq5_text/text_pairs.jsonl"):
    r = json.loads(line)
    prompts.append(r["prompt"]); single[r["idx"]] = r["single_text"]; poe[r["idx"]] = r["poe_text"]
idxs = list(range(len(prompts)))
textmap = {"raw": lambda i: prompts[i], "single": lambda i: single[i], "poe": lambda i: poe[i]}

# deterministic flat work list, sharded across tasks
units = [(cfg, cond, i, seed)
         for cfg in CFGS for cond in CONDS for i in idxs for seed in SEEDS]
mine = [u for k, u in enumerate(units) if k % WORLD == RANK]
print(f"rank {RANK}/{WORLD}: {len(mine)}/{len(units)} work units", flush=True)

pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True)

made = 0
for cfg, cond, i, seed in mine:
    d = OUT / cond / f"cfg_{cfg:g}"; d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{i:03d}_s{seed}.png"
    if fp.exists():
        continue
    g = torch.Generator("cuda").manual_seed(seed)
    im = pipe(prompt=textmap[cond](i), num_inference_steps=STEPS,
              guidance_scale=cfg, generator=g).images[0]
    im.save(fp); made += 1
    if made % 50 == 0:
        print(f"rank {RANK}: +{made}", flush=True)
print(f"DONE_FID_GEN rank {RANK} made {made}", flush=True)
