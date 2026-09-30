# Big run — regenerate all dissertation assets

One coordinated pass that produces every figure and metric the dissertation uses:
RQ1–RQ6, the diagnostics, cross-backbone comparison, the Product-of-Experts
capstone (RQ5, 100 prompts, shared-scene construction), the steering analysis, and
the anchor-weight sweep.

## Prerequisites
- Run every command from the repo root (scripts rely on `SLURM_SUBMIT_DIR`).
- Rewrite caches present (`outputs/rewrite_cache/{ar_llama3.1,llada}.json`); if
  absent the first submit builds them via the warmup jobs.
- Ollama tarball at `~/ollama-dist` (decomposition + AR rewrites + diagnostics).
- `config/experiment.yaml` sets the primary backbone; scripts also take it per run.

## Submission sequence
The cluster caps submitted jobs (~6) and each backbone's RQ chain is dependency-
linked, so submit in waves rather than all at once.

1. Primary backbone, full pipeline (RQ1–RQ6 + cross-backbone compare):
       bash scripts/submit_hpc.sh --backbones sdxl
2. Second backbone for the cross-backbone contrast (RQ1+RQ2 minimum):
       bash scripts/submit_hpc.sh --backbones sd21
   The final prompt-compare job reads outputs/sd21 + outputs/sdxl.
3. Diagnostics (denoising trajectory, CFG-stability, LLaDA language-layer
   trajectory, and the RQ1 embedding export that feeds the PCA figure):
       bash scripts/submit_hpc.sh --diagnostics --backbones sdxl
       bash scripts/submit_hpc.sh --diagnostics --backbones sd21
4. Anchor-weight sweep (text-level; pins the w=1 balance claim):
       sbatch scripts/submit_sweep.sh
5. Construction ablation on the full set (optional; firms Table 4.5 at n=18):
       sbatch scripts/submit_ablation.sh

Monitor: squeue -u $USER ; tail -f logs/slurm/*.out

## What produces each figure
All figures land in outputs/<backbone>/plots/ (grids under plots/figures/),
written by evaluation.plotting.generate_all_plots, which each RQ job and the
diagnostics job re-run over on-disk artifacts.

  RQ1 counts             rq1_attr_count            RQ1 run
  RQ1 separation gain    rq1_separation_gain       RQ1 run
  RQ1 embedding PCA      rq1_embedding_pca         diagnostics (embedding export)
  RQ2 bars               rq2_bar_*                 RQ2 run
  RQ2 FID                rq2_fid                   RQ2 run
  RQ2 qualitative grids  figures/rq2_grid_<set>    RQ2 run
  RQ3 CFG sweeps         rq3_cfg_*                 RQ3 run
  RQ3 stability          rq3_stability             RQ3 run
  RQ4 mechanism deltas   rq4_mechanism_deltas      RQ4 run
  RQ4 trajectory         trajectory_clip_score     diagnostics (see gap)
  RQ5 4-condition grid   figures/rq5_poe_grid      RQ5 run
  RQ5 text comparison    rq5_text_compare          RQ5 run
  RQ6 CFG saturation     rq6_cfg_saturation        RQ6 run
  RQ6 cost vs steps      rq6_time                  RQ6 run
  RQ6 tunability grid    rq6_image_grid            RQ6 run
  RQ6 steering           rq6_steerability          RQ6 run (new)
  Inference timing       rewrite_timing            any run
  Cross-backbone         compare_*                 prompt-compare job

## Copying figures into the dissertation
writeup/ is git-ignored and the .tex includes from writeup/images/. After the run,
copy the vector figures across (names already match) and uncomment the matching
\includegraphics lines:

  cd ~/compositional-prompt-refinement
  for f in rq1_attr_count rq1_separation_gain rq1_embedding_pca \
           rq3_stability rq4_mechanism_deltas \
           rq6_cfg_saturation rq6_time rq6_image_grid rq6_steerability \
           rewrite_timing; do
    cp -f outputs/sdxl/plots/$f.pdf writeup/images/ 2>/dev/null
  done
  cp -f outputs/sdxl/plots/figures/rq5_poe_grid.pdf writeup/images/ 2>/dev/null

## Remaining gaps (need attention, not just a run)
- RQ4 corrected trajectory: confirm the trajectory diagnostic scores against the
  ORIGINAL prompt, not each pipeline's own rewrite, before citing the raw-vs-
  rewrite gap. The AR≈LLaDA overlap is safe regardless.
- §4.5.2 (image-level PoE) and §4.7 (cross-backbone) prose: written once the RQ5
  capstone and the sd21 run land.
- The anchor sweep is text-level; an image-level weight sweep is a further run.
