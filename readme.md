# Diffusion-Based Refinement of Textual Conditioning

Empirical investigation into whether diffusion-based prompt refinement produces more compositionally faithful conditioning signals for text-to-image generation than raw prompting or autoregressive rewriting.

MSc research by Oratile Nailana, University of the Witwatersrand, supervised by Dr. Devon Jarvis.

---

## Overview

Text-to-image diffusion models are sensitive to prompt quality. Natural language prompts are often underspecified, leading to attribute misbinding, relational inconsistencies, and object entanglement in generated images. This project evaluates whether refining prompts in language space — before they reach the image generation backbone — can reduce these failures.

Three conditioning strategies are compared under a frozen SD 2.1 backbone:

| Strategy | Pipeline name | Description |
|----------|---------------|-------------|
| Raw prompting | `raw_clip` | User prompt encoded directly via CLIP-H/14 |
| Autoregressive rewrite | `ar_clip` | Llama 3.1 (via Ollama, temp=0) expands the prompt, then CLIP-H/14 encodes it |
| Diffusion refinement | `llada_clip` | LLaDA-8B-Instruct iteratively unmasks the prompt into a richer description, then CLIP-H/14 encodes it |

All three pipelines share the same encoder (CLIP-H/14, `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`) and the same frozen T2I backbone (SD 2.1 at 768×768). All variation is in the conditioning pathway only.

> **Long-CLIP status:** A Long-CLIP-L (248-token) ablation path exists in the codebase (`encoders/longclip_encoder.py`, `generation/t2i_runner.py::EmbeddingProjector`), but it is disabled in `config/experiment.yaml` pending fine-tuning of the linear EmbeddingProjector (CLIP-H 1024-dim → Long-CLIP 768-dim, seq 77→248). Re-enable by adding `longclip` to the `encoders` list in the config once the projector checkpoint is available.

> **LLaDA / GENIE note:** The proposal refers to GENIE (Lin et al., 2023) as the diffusion language model. LLaDA-8B-Instruct (masked diffusion, iterative token unmasking) is used here as the closest publicly available model with the same generative mechanism. `pipeline/genie_refine.py` aliases `LLaDAPipeline` for swap compatibility.

---

## Research Questions

- **RQ1** — Does diffusion-based refinement produce more semantically structured conditioning signals than raw prompts? (text + embedding analysis, no image generation)
- **RQ2** — Does improved conditioning reduce compositional errors? (full generation + CLIPScore, FID, BLIP-2 VQA)
- **RQ3** — Does prompt refinement reduce sensitivity to classifier-free guidance scale?
- **RQ4** — Does the diffusion mechanism produce stronger improvements than autoregressive rewriting under matched expansion instructions?

---

## Project Structure

```
.
├── config/
│   ├── experiment.yaml          # All hyperparameters — model paths, CFG scales, seeds, eval sets
│   └── prompts.yaml             # Benchmark prompt sets (color_binding, spatial_relations, …)
│
├── pipeline/
│   ├── base.py                  # Abstract ConditioningPipeline + EncodingResult dataclass
│   ├── raw.py                   # raw_clip: c = f_enc(t)
│   ├── ar_rewrite.py            # ar_clip:  c = f_enc(g_AR(t))
│   ├── llada_refine.py          # llada_clip: c = f_enc(g_LLaDA(t))
│   └── genie_refine.py          # Alias for LLaDAPipeline (swap when GENIE checkpoint is available)
│
├── encoders/
│   ├── base.py                  # Abstract TextEncoder interface
│   ├── clip_encoder.py          # CLIP-H/14 (seq=77, hidden_dim=1024) — active
│   └── longclip_encoder.py      # Long-CLIP-L (seq=248, hidden_dim=768) — disabled (see above)
│
├── rewriters/
│   ├── base.py                  # Abstract PromptRewriter interface
│   ├── ollama_rewriter.py       # AR baseline: Llama 3.1 via local Ollama, temp=0, disk cache
│   └── llada_rewriter.py        # Diffusion rewriter: LLaDA-8B-Instruct, temp=0, disk cache
│
├── generation/
│   └── t2i_runner.py            # SD 2.1 wrapper (768×768, v-prediction, DPMSolverMultistep)
│                                #   Injects conditioning via prompt_embeds; optional EmbeddingProjector
│
├── evaluation/
│   ├── metrics.py               # CLIPScorer, FIDScorer, AttributeBindingScorer,
│   │                            #   RelationAccuracyScorer (latter two share one BLIP-2 instance)
│   ├── embedding_analysis.py    # RQ1: semantic density, cosine embedding separation
│   ├── cfg_sensitivity.py       # RQ3: CFG sweep, compositional stability
│   └── qualitative.py           # Rater annotation loader, Cohen's κ
│
├── experiments/
│   ├── run_experiment.py        # Main CLI entry point
│   ├── rq1_conditioning.py      # Semantic density + embedding separation
│   ├── rq2_compositional.py     # Generation benchmark (two-phase: generate → unload → score)
│   ├── rq3_cfg_sensitivity.py   # CFG sweep
│   └── rq4_mechanism.py         # AR vs LLaDA head-to-head
│
├── utils/
│   ├── seed.py                  # set_seed(), get_generator()
│   ├── logging.py               # W&B integration
│   ├── prompt_io.py             # load_prompts(), expansion ratio logging
│   └── truncation_monitor.py    # Per-pipeline truncation event tracking
│
├── scripts/
│   ├── setup_linux.sh           # Full environment setup (venv, deps, Ollama, HF login)
│   ├── prepare_prompts.sh       # Download and format T2I-CompBench++ sets
│   ├── submit_hpc.sh            # SLURM job templates
│   └── run_local.sh             # Single-GPU dry run for development
│
├── test_pipeline.py             # Smoke test (CPU-safe, no GPU required)
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python **3.10–3.12** (`tokenizers` and `safetensors` do not yet support 3.14)
- CUDA-capable GPU (recommended: ≥ 16 GB VRAM for LLaDA + SD 2.1 simultaneously)
- [Ollama](https://ollama.com) installed locally for the AR rewrite condition
- A [HuggingFace](https://huggingface.co/settings/tokens) account token (SD 2.1, CLIP-H, LLaDA, BLIP-2)

### Automated setup (Linux / HPC)

```bash
bash scripts/setup_linux.sh
```

The script creates a `venv/`, installs PyTorch + all dependencies, starts Ollama, and logs you in to HuggingFace. Run it once before any experiment.

### Manual setup

```bash
# 1. Virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 2. PyTorch with CUDA (pick your version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8
pip install torch torchvision                                                        # CPU only

# 3. All other dependencies
pip install -r requirements.txt

# 4. HuggingFace login (required for SD 2.1 and LLaDA weights)
huggingface-cli login
```

### Ollama (AR baseline)

```bash
ollama serve
```

`llama3.1` is pulled automatically on first use — no manual `ollama pull` required. `OllamaRewriter` streams pull progress and logs it at INFO level.

### LLaDA (diffusion rewriter)

`GSAI-ML/LLaDA-8B-Instruct` (~16 GB) is downloaded from HuggingFace automatically on first use. Requires CUDA. Running on CPU is technically possible but takes hours per prompt — only use CPU for debugging.

### Environment variables (`.env`)

Copy `.env.example` to `.env` and fill in your tokens:

```bash
WANDB_API_KEY=          # https://wandb.ai/authorize
WANDB_OFFLINE=false     # set true on HPC nodes without internet

HF_TOKEN=               # https://huggingface.co/settings/tokens
```

`setup_linux.sh` reads `HF_TOKEN` from the environment and runs `huggingface-cli login` automatically. If `HF_TOKEN` is not set it will prompt interactively.

---

## Running Experiments

### Smoke test (no GPU required)

```bash
python test_pipeline.py
```

Verifies CLIP encoding on CPU, reports Ollama and CUDA availability.

### Dry run (CPU-safe, fast)

```bash
python experiments/run_experiment.py --rq 1 --dry-run
python experiments/run_experiment.py --rq 2 --dry-run
```

Runs 3 prompts through `raw_clip` and `ar_clip` (LLaDA skipped) with 2 denoising steps. BLIP-2 and FID scorers are replaced with stubs returning 0.0 to avoid OOM on memory-constrained machines. CLIPScore is still loaded and gives a real signal. Use this to verify the full stack before committing GPU hours.

### Full runs

```bash
python experiments/run_experiment.py --rq 1    # Conditioning structure analysis (text + embeddings, no image gen)
python experiments/run_experiment.py --rq 2    # Compositional benchmark (CLIPScore, FID, BLIP-2 VQA)
python experiments/run_experiment.py --rq 3    # CFG sensitivity sweep
python experiments/run_experiment.py --rq 4    # AR vs LLaDA mechanism comparison
python experiments/run_experiment.py --rq all  # All RQs sequentially
```

### CLI overrides

```bash
python experiments/run_experiment.py --rq 2 --seed 456
python experiments/run_experiment.py --rq 2 --cfg 5.0
python experiments/run_experiment.py --rq 3 --config sampling_steps=20
python experiments/run_experiment.py --rq 2 --no-wandb           # disable W&B
python experiments/run_experiment.py --rq 2 --pipeline ar_clip   # single pipeline (SLURM array use)
```

### HPC / SLURM

```bash
bash scripts/submit_hpc.sh
```

Submits separate SLURM jobs for each RQ. The `--pipeline` flag allows array jobs to run each pipeline in parallel. See the script for partition and memory settings.

---

## Configuration

`config/experiment.yaml` controls all parameters. Key fields:

```yaml
# Models
t2i_model: sd2-community/stable-diffusion-2-1   # 768×768 v-prediction model
clip_model: laion/CLIP-ViT-H-14-laion2B-s32B-b79K
llada_model: GSAI-ML/LLaDA-8B-Instruct
ollama_model: llama3.1

# Generation
image_size: 768          # SD 2.1 is a v-prediction model trained at 768×768
                         # (use 512 only for stable-diffusion-2-1-base, which is ε-prediction)
sampling_steps: 100
default_cfg_scale: 7.5
cfg_scales: [1.0, 3.0, 5.0, 7.5, 10.0]   # RQ3 sweep

# Active encoder (Long-CLIP disabled pending EmbeddingProjector fine-tuning)
encoders:
  - clip
  # - longclip

# Rewrite cache — all RQs share the same rewrites for experimental integrity
rewrite_cache_dir: outputs/rewrite_cache/

# Seeds (five runs for variance estimation)
seeds: [42, 123, 456, 789, 1024]

# Tracking
wandb_project: prompt-pipeline
output_dir: outputs/
```

`config/prompts.yaml` contains real T2I-CompBench++ validation prompts (Huang et al.) plus one auxiliary set:

| Set | Prompts | Source | Used in |
|-----|---------|--------|---------|
| `color_binding` | 240 | T2I-CompBench++ `color_val.txt` | RQ1, RQ2, RQ4 |
| `shape_binding` | 50 | T2I-CompBench++ `shape_val.txt` | RQ1, RQ2 |
| `texture_binding` | 50 | T2I-CompBench++ `texture_val.txt` | RQ1, RQ2 |
| `spatial_relations` | 102 | T2I-CompBench++ `spatial_val.txt` | RQ1, RQ2, RQ4 |
| `non_spatial` | 50 | T2I-CompBench++ `non_spatial_val.txt` | RQ2 |
| `cats_dogs` | 8 | hand-crafted auxiliary | RQ2, RQ4 |
| `rq3_sweep` | 25 | 5 per category (cross-set sample) | RQ3 |

---

## Evaluation Metrics

| Metric | Implementation | RQ |
|--------|---------------|----|
| CLIPScore | CLIP cosine similarity (image vs prompt embedding) | RQ1, RQ2, RQ4 |
| FID | Fréchet Inception Distance vs first pipeline's images as reference | RQ2, RQ4 |
| Attribute Binding Accuracy | BLIP-2 VQA: "Is there a `<attr>` `<obj>`?" | RQ2, RQ4 |
| Relation Accuracy | BLIP-2 VQA: "Is the `<A>` `<rel>` the `<B>`?" | RQ2, RQ4 |
| Compositional Stability | Variance of metrics across CFG scales | RQ3 |
| Semantic Density | Attribute/relation token count in rewritten vs raw prompts | RQ1 |
| Embedding Separation | Mean pairwise cosine distance across prompt set | RQ1 |

`TruncationMonitor` tracks truncation events (tokens lost when rewritten prompts exceed 77 tokens) separately per pipeline, letting the analysis distinguish encoder bottleneck effects from refinement quality effects.

---

## Output Structure

```
outputs/
├── rewrite_cache/
│   ├── ar_llama3.1.json         # {raw_prompt: rewrite} for Ollama/Llama
│   └── llada.json               # {raw_prompt: rewrite} for LLaDA
│
├── rq1/
│   ├── rq1_summary.txt          # Aggregate density + separation stats per pipeline
│   ├── trace_raw_clip.jsonl     # Per-prompt: raw→rewrite, token counts, semantic density
│   ├── trace_ar_clip.jsonl
│   └── trace_llada_clip.jsonl
│
├── rq2/
│   ├── rq2_summary.txt          # Aggregate CLIPScore, FID, VQA per pipeline × set
│   └── {pipeline}/{set}/
│       ├── prompt_000.png       # Generated images
│       ├── prompt_001.png
│       └── trace.jsonl          # Per-prompt: raw→rewrite→image_path→all scores
│
├── rq3/ …                       # CFG sweep results
└── rq4/ …                       # AR vs LLaDA delta metrics + trace.jsonl
```

`trace.jsonl` files are the primary traceability artifact: each line is a JSON record linking a raw prompt to its rewrite, token counts, truncation flag, generated image path, and all per-image scores. This lets you audit any individual result end-to-end.

---

## Key Design Decisions

**`prompt_embeds` injection.** `T2IRunner` bypasses SD 2.1's internal text encoder and passes conditioning tensors directly via diffusers' `prompt_embeds` parameter. The conditioning pathway is entirely under our control without touching the backbone weights.

**SD 2.1 is a v-prediction model.** `sd2-community/stable-diffusion-2-1` was trained with v-prediction (`prediction_type: v_prediction`) at 768×768. The community repo is missing this field in its scheduler config, so loading the pipeline naively gives epsilon-prediction — blurry, incoherent output. `T2IRunner._load()` explicitly overrides the scheduler to `DPMSolverMultistepScheduler(prediction_type="v_prediction", use_karras_sigmas=True)`. If you switch to `stabilityai/stable-diffusion-2-1-base`, use 512×512 and remove the override — that checkpoint uses epsilon-prediction.

**Disk-backed rewrite cache.** `OllamaRewriter` and `LLaDARewriter` each maintain a `{prompt → rewrite}` JSON cache at `outputs/rewrite_cache/`. Every RQ reads from this cache, so all RQs share the exact same rewrites — meaning the conditioning vectors analysed in RQ1 are identical to those that generated the RQ2 images. Without this, each RQ would independently call the LLM and potentially produce different expansions despite `temperature=0`. The cache also avoids redundant LLM inference across runs.

**Two-phase RQ2.** Holding SD 2.1 (~3.5 GB), LLaDA (~16 GB), and BLIP-2 (~15 GB float32 on CPU) in memory simultaneously exceeds most machines' capacity. RQ2 therefore runs in two phases: Phase 1 generates all images and caches encoded results; Phase 2 calls `runner.unload()` (frees SD 2.1 VRAM) and then scores with BLIP-2 and CLIP. Controlled by `unload_runner_before_scoring=True` in `run_rq2()`.

**Matched expansion instructions.** `OllamaRewriter` and `LLaDARewriter` use the exact same `_EXPANSION_INSTRUCTION` string. The only variable in RQ4 is the generative mechanism (left-to-right autoregressive vs masked diffusion iterative unmasking). Do not change one without updating the other.

**Shared conditioning interface.** All three pipelines implement `ConditioningPipeline.encode(prompt) → EncodingResult`. `T2IRunner` and all scorers are agnostic to which strategy they receive, making it straightforward to add new pipelines or encoders without touching experiment code.

**Frozen backbone throughout.** SD 2.1 weights are never updated. All experimental variation comes from the conditioning pathway alone.

**BLIP-2 shared instance.** `AttributeBindingScorer` and `RelationAccuracyScorer` both need BLIP-2 flan-t5-xl. A module-level `_BLIP2_CACHE` in `evaluation/metrics.py` ensures the ~15 GB model is loaded only once regardless of how many scorer instances are created.

---

## Hardware Requirements

| Component | VRAM / RAM | Notes |
|-----------|------------|-------|
| SD 2.1 (generation) | ~3.5 GB VRAM | float16 on CUDA; float32 on CPU (slow) |
| CLIP-H/14 (encoder) | ~1.5 GB VRAM | float16 on CUDA, float32 on CPU |
| LLaDA-8B (rewriter) | ~16 GB VRAM | bfloat16; CPU fallback takes hours per prompt |
| BLIP-2 flan-t5-xl (scoring) | ~8 GB VRAM float16 / ~15 GB RAM float32 | CPU run risks OOM on 16 GB machines |
| Ollama / Llama 3.1 (AR rewriter) | ~4 GB RAM | Managed by Ollama; CPU is usable |

RQ2 with LLaDA requires a GPU with ≥ 16 GB VRAM for comfortable single-card execution. On HPC, submit `llada_clip` as a separate array task (`--pipeline llada_clip`) to a high-memory partition.

---

## Citation

If you build on this work, please cite the underlying proposal:

```
Nailana, O. (2026). Diffusion-Based Refinement of Textual Conditioning for Improved
Generative Modeling. MSc Research Proposal, University of the Witwatersrand.
Supervised by Dr. Devon Jarvis.
```

Key references: LLaDA (Nie et al., 2025), GENIE (Lin et al., 2023), T2I-CompBench++ (Huang et al., 2024), Long-CLIP (Zhang et al., 2024), Stable Diffusion (Rombach et al., 2022), BLIP-2 (Li et al., 2023).
