"""RQ5 x RQ6 -- generate SDXL images across a CFG grid for FID evaluation.

Saves PNGs to outputs/sdxl/fid/<pipeline>/cfg_<cfg>/<idx>_s<seed>.png.
Resumable: skips any image already written, so a walltime cap just needs a resubmit.
No manual upcast_vae() -- the SDXL pipeline auto-upcasts its fp16 VAE for the
final decode (manual upcast breaks it: Half vs float).
"""
import json
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
STEPS = 50
CFGS  = [3.0, 7.5, 15.0, 30.0, 50.0]   # default -> deep overshoot
SEEDS = [42, 43]                        # 100 prompts x 2 seeds = 200 imgs / cell
OUT   = Path("outputs/sdxl/fid")

prompts, single, poe = [], {}, {}
for line in open("outputs/sdxl/rq5_text/text_pairs.jsonl"):
    r = json.loads(line)
    prompts.append(r["prompt"]); single[r["idx"]] = r["single_text"]; poe[r["idx"]] = r["poe_text"]
idxs = list(range(len(prompts)))
print(f"prompts: {len(idxs)}", flush=True)

pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True)
conditions = {"raw": lambda i: prompts[i], "single": lambda i: single[i], "poe": lambda i: poe[i]}

for cfg in CFGS:
    for cond, textfn in conditions.items():
        d = OUT / cond / f"cfg_{cfg:g}"; d.mkdir(parents=True, exist_ok=True)
        made = 0
        for i in idxs:
            for seed in SEEDS:
                fp = d / f"{i:03d}_s{seed}.png"
                if fp.exists():
                    continue
                g = torch.Generator("cuda").manual_seed(seed)
                im = pipe(prompt=textfn(i), num_inference_steps=STEPS,
                          guidance_scale=cfg, generator=g).images[0]
                im.save(fp); made += 1
        print(f"[cfg={cfg:g} {cond}] +{made} (dir now {len(list(d.glob('*.png')))})", flush=True)
print("DONE_FID_GEN", flush=True)
