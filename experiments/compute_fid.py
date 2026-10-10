"""Compute FID of each (pipeline, cfg) generated set vs a COCO reference dir.

Usage: python experiments/compute_fid.py <ref_dir>
Writes outputs/sdxl/fid/fid_results.json : {"<cond>|<cfg>": {fid, n_gen, n_ref}}.
Equal n per cell (200), so absolute FID is biased high but CROSS-CELL comparison
(does FID rise with cfg, faster for some pipeline?) is valid.
"""
import sys, json, glob
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

REF_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fid_ref/val2017")
GEN = Path("outputs/sdxl/fid")
CFGS = [3.0, 7.5, 15.0, 30.0, 50.0]
CONDS = ["raw", "single", "poe"]
dev = "cuda" if torch.cuda.is_available() else "cpu"

def load_uint8(paths):
    ims = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB").resize((299, 299))
            ims.append(torch.from_numpy(np.array(im)).permute(2, 0, 1))
        except Exception as e:
            print("skip", p, e, flush=True)
    return torch.stack(ims).to(torch.uint8) if ims else None

ref_paths = sorted(glob.glob(str(REF_DIR / "*.jpg"))) + sorted(glob.glob(str(REF_DIR / "*.png")))
print("ref images:", len(ref_paths), flush=True)
ref = load_uint8(ref_paths)
print("ref tensor:", tuple(ref.shape), flush=True)

results = {}
for cond in CONDS:
    for cfg in CFGS:
        gp = sorted(glob.glob(str(GEN / cond / f"cfg_{cfg:g}" / "*.png")))
        if not gp:
            continue
        gen = load_uint8(gp)
        fid = FrechetInceptionDistance(feature=2048, normalize=False).to(dev)
        for k in range(0, ref.shape[0], 64):
            fid.update(ref[k:k+64].to(dev), real=True)
        for k in range(0, gen.shape[0], 64):
            fid.update(gen[k:k+64].to(dev), real=False)
        val = float(fid.compute())
        results[f"{cond}|{cfg:g}"] = {"fid": val, "n_gen": gen.shape[0], "n_ref": ref.shape[0]}
        print(f"{cond:7s} cfg={cfg:<4g} FID={val:8.2f}  (n_gen={gen.shape[0]})", flush=True)
        del fid
        if dev == "cuda":
            torch.cuda.empty_cache()

json.dump(results, open(GEN / "fid_results.json", "w"), indent=2)
print("DONE_FID_COMPUTE", flush=True)
