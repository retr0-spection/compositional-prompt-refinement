# Experiment registry

The authoritative map of every experiment: what it answers, its code, how it is
launched, where its output lands, and which figure it feeds.

**Convention (one line to remember):**
> one experiment = one script = one SLURM wrapper = one output dir = one figure
>
> `experiments/<name>.py`  <->  `scripts/_slurm_<name>.sh`
> `outputs/<backbone>/<name>/`  <->  figure `<name>.png`

`<bb>` = backbone (`sdxl`, `sd21`, ...). The SDXL capstone lives under `outputs/sdxl/`.

## Registry

| Experiment | RQ | What it answers | Script | SLURM wrapper | Output dir | Figure(s) | Status |
|---|---|---|---|---|---|---|---|
| rq1_conditioning | RQ1 | Conditioning strategies: attribute count & embedding separation | `experiments/rq1_conditioning.py` | `_slurm_task.sh` | `outputs/<bb>/rq1` | rq1_attr_count, rq1_separation_gain, rq1_embedding_pca | done |
| export_embeddings | RQ1 | Export text-encoder embeddings for the PCA figure | `experiments/export_embeddings.py` | `_slurm_task.sh` | `outputs/<bb>/rq1` | rq1_embedding_pca | done |
| rq2_compositional | RQ2 | Compositional fidelity across conditioning methods | `experiments/rq2_compositional.py` | `_slurm_task.sh` | `outputs/<bb>/rq2` | rq2_bar_*, rq2_fid | done |
| rq3_cfg_sensitivity | RQ3 | CFG sensitivity of the metrics across methods | `experiments/rq3_cfg_sensitivity.py` | `_slurm_task.sh` | `outputs/<bb>/rq3` | rq3_cfg_*, rq3_stability | done |
| rq4_mechanism | RQ4 | Mechanism: per-step denoising deltas (null, reframed) | `experiments/rq4_mechanism.py` | `_slurm_task.sh` | `outputs/<bb>/rq4` | rq4_mechanism_deltas | done |
| trajectory | RQ4 | Denoising CLIP trajectory vs original prompt (orig. diagnostic) | `experiments/trajectory.py` | `_slurm_diagnostics.sh` | `outputs/<bb>/trajectory` | trajectory_clip_score | done |
| rq5_poe | RQ5 | Shared-scene PoE image gen: 4 conditions x 100 prompts | `experiments/rq5_poe.py` | `_slurm_task.sh` (job 65826) | `outputs/sdxl/rq5` | rq5_image_bars, rq5_poe_grid | done (10h18m) |
| rq5_text_compare | RQ5 | Text-stage coverage: single vs PoE rewrites | `experiments/rq5_text_compare.py` | `_slurm_compare.sh` | `outputs/sdxl/rq5_text` | rq5_text_coverage, rq5_length_dist | done |
| rescore_relations | RQ5 | Re-score relation accuracy w/ fixed parser on saved images | `experiments/rescore_relations.py` | `_slurm_rescore.sh` (job 65991) | `outputs/sdxl/rq5/relation_rescore.jsonl` | feeds rq5_image_bars | done |
| rq5_poe_ablation | RQ5 | PoE ablation: expert weights / fluency anchor | `experiments/rq5_poe_ablation.py` | `submit_ablation.sh` | `outputs/sdxl/rq5_ablation` | (ablation bars) | optional |
| rq5_clip_trajectory | RQ5xRQ6 | Image-space CLIP trajectory during denoising, CFG {7.5,15} | `experiments/rq5_clip_trajectory.py` | `_slurm_rq5_clip_trajectory.sh` | `outputs/sdxl/rq5_clip_trajectory` | rq5_clip_trajectory.png | queued |
| rq5_cfg_robustness | RQ5xRQ6 | Pipeline CLIP robustness vs guidance, grid 3->50 (overshoot) | `experiments/rq5_cfg_robustness.py` | `_slurm_rq5_cfg_robustness.sh` | `outputs/sdxl/rq5_cfg_robustness` | rq5_cfg_robustness.png | queued |
| tunability (RQ6) | RQ6 | Steerability: expert-weight / CFG tunability | `experiments/tunability.py` | `_slurm_task.sh` | `outputs/<bb>/rq6` | rq6_cfg_saturation, rq6_steerability, rq6_image_grid | done |
| llada_trajectory | diag | LLaDA text-diffusion unmasking trajectory | `experiments/llada_trajectory.py` | `_slurm_diagnostics.sh` | `outputs/llada_trajectory` | (unmasking diag) | done |
| cfg_stability | diag | CFG-stability across denoising STEPS (per-step smoothness, `--set` driven) | `experiments/cfg_stability.py` | `_slurm_diagnostics.sh` | `outputs/<bb>/diagnostics` | (trajectory smoothness) | available |
| run_experiment | infra | Main driver: dispatches RQ1-RQ6 by `--set` / `--backbone` | `experiments/run_experiment.py` | `_slurm_task.sh` | `outputs/<bb>/<set>` | - | infra |

## The three CFG experiments -- do not confuse them

They sound alike but measure different things:

- **rq3_cfg_sensitivity** -- variance of the *final image quality* across CFG scales (RQ3).
- **cfg_stability** -- smoothness of the *per-step denoising trajectory* at each CFG scale
  (does strong guidance make the step-to-step path thrash?). `--set` driven diagnostic.
- **rq5_cfg_robustness** -- *final CLIPScore vs CFG* across the grid 3->50, compared across
  pipelines (raw / single / PoE). Tests whether refinement sustains alignment into the
  oversaturation tail. The proposal's "CLIPScore stability with increasing CFG" metric.

## Naming resolved this session (collision fix)

`cfg_stability.py` and `clip_trajectory.py` were reused for two new RQ5xRQ6 sweeps, which
overwrote / overlapped existing experiments. Resolved:
- `cfg_stability.py` restored to the ORIGINAL per-step smoothness diagnostic (from git 6fd8988).
- `rq5_cfg_robustness.py` = NEW final-CLIP-vs-CFG sweep (grid 3->50). (was `cfg_stability.py`)
- `rq5_clip_trajectory.py` = NEW image-space CLIP trajectory. (was `clip_trajectory.py`)

## Status legend

done = run complete | queued = submitted, awaiting node | optional = not on critical path |
available = ready to run, not scheduled | infra = shared driver, not a standalone experiment.
