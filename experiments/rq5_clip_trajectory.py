"""RQ5 x RQ6 -- image-space CLIP trajectory during denoising (CFG sweep).

For a small prompt sample, generate with SDXL and, at a cadence through the
reverse-diffusion loop, decode the current latent to a preview image and score
CLIPScore against the ORIGINAL prompt. Shows how prompt alignment emerges
(sigmoid-like) and whether conditioning (raw / single / PoE) changes the shape,
swept across guidance scales (default vs high).

Distinct from experiments/trajectory.py (RQ4 denoising diag) and
experiments/llada_trajectory.py (LLaDA text-diffusion unmasking).

Output: outputs/sdxl/rq5_clip_trajectory/clip_vs_step.jsonl
        {cond, cfg, idx, step, frac, clip}

Note: decodes the current latent (noisy early) - an approximate preview, not the
model x0 estimate. Diagnostic only; 50 steps (vs the 100-step main run).
"""
import json
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline
from evaluation.metrics import CLIPScorer

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
STEPS = 50
CFGS = [7.5, 15.0]  # default vs high-guidance (steerability regime)
SEED = 42
EVERY = 5
N_PROMPTS = 10
OUT = Path("outputs/sdxl/rq5_clip_trajectory")
OUT.mkdir(parents=True, exist_ok=True)

prompts, single, poe = [], {}, {}
for line in open("outputs/sdxl/rq5_text/text_pairs.jsonl"):
    r = json.loads(line)
    prompts.append(r["prompt"]); single[r["idx"]] = r["single_text"]; poe[r["idx"]] = r["poe_text"]
idxs = list(range(min(N_PROMPTS, len(prompts))))

print("loading SDXL...", flush=True)
pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True)
pipe.upcast_vae()
clip = CLIPScorer()

records = []
conditions = {"raw": lambda i: prompts[i], "single": lambda i: single[i], "poe": lambda i: poe[i]}

for cfg in CFGS:
    for cond, textfn in conditions.items():
        print("[cfg=" + str(cfg) + " " + cond + "]", flush=True)
        for i in idxs:
            orig = prompts[i]
            def cb(pp, step, t, kw, _orig=orig, _cond=cond, _i=i, _cfg=cfg):
                if step % EVERY == 0 or step == STEPS - 1:
                    lat = kw["latents"].to(pp.vae.dtype) / pp.vae.config.scaling_factor
                    with torch.no_grad():
                        im = pp.vae.decode(lat, return_dict=False)[0]
                    im = pp.image_processor.postprocess(im, output_type="pil")[0]
                    records.append({"cond": _cond, "cfg": float(_cfg), "idx": _i,
                                    "step": int(step), "frac": round((step + 1) / STEPS, 3),
                                    "clip": float(clip.score(im, _orig))})
                return kw
            g = torch.Generator("cuda").manual_seed(SEED)
            pipe(prompt=textfn(i), num_inference_steps=STEPS, guidance_scale=cfg,
                 generator=g, callback_on_step_end=cb,
                 callback_on_step_end_tensor_inputs=["latents"])

with open(OUT / "clip_vs_step.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print("DONE_RQ5_CLIP_TRAJECTORY", len(records), flush=True)
