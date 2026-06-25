# Diffusion-Based Refinement of Textual Conditioning

Empirical investigation into whether diffusion-based prompt refinement produces more compositionally faithful conditioning signals for text-to-image generation than raw prompting or autoregressive rewriting.

Based on the MSc research proposal by Oratile Nailana, University of the Witwatersrand, supervised by Dr. Devon Jarvis.

---

## Overview

Text-to-image diffusion models are sensitive to prompt quality. Natural language prompts are often underspecified, leading to attribute misbinding, relational inconsistencies, and object entanglement in generated images. This project evaluates whether refining prompts in language space — before they reach the image generation backbone — can reduce these failures.

Three conditioning strategies are compared under a frozen T2I backbone:

| Strategy | Symbol | Description |
|----------|--------|-------------|
| Raw prompting | `c_raw` | User prompt encoded directly |
| Autoregressive rewrite | `c_ar` | Llama 3 (via Ollama) expands the prompt |
| Diffusion refinement | `c_ref` | LLaDA-8B-Instruct iteratively denoises the prompt into a richer description |

Each strategy is evaluated with both **CLIP** (77-token window) and **Long-CLIP** (248-token window) encoders, yielding a **2×3 experimental design**. The T2I backbone (Stable Diffusion 2.1) is frozen throughout — all variation is in the conditioning pathway.

> **Implementation note:** The proposal refers to GENIE (Lin et al., 2023) as the diffusion language model. LLaDA-8B-Instruct (masked diffusion, iterative token unmasking) is used here as a publicly available model with the same generative mechanism. `pipeline/genie_refine.py` aliases `LLaDAPipeline` and can be swapped for a GENIE checkpoint when available.

---

## Research Questions

- **RQ1** — Does diffusion-based refinement produce more semantically structured conditioning signals than raw prompts?
- **RQ2** — Does improved conditioning reduce compositional errors (attribute misbinding, relational inconsistency)?
- **RQ3** — Does prompt refinement reduce sensitivity to classifier-free guidance scale?
- **RQ4** — Does the diffusion mechanism produce stronger improvements than autoregressive rewriting under matched expansion instructions?

---

## Project Structure

```
.
├── config/
│   ├── experiment.yaml          # All hyperparameters: CFG scales, seeds, model paths, eval sets
│   └── prompts.yaml             # Benchmark prompt sets (attribute_binding, spatial_relations,
│                                #   celebA, cats_dogs, rq3_sweep)
│
├── pipeline/
│   ├── base.py                  # Abstract ConditioningPipeline + EncodingResult
│   ├── raw.py                   # RawPipeline: c_raw = f_enc(t)
│   ├── ar_rewrite.py            # ARPipeline: c_ar = f_enc(g_AR(t))
│   ├── llada_refine.py          # LLaDAPipeline: c_ref = f_enc(g_LLaDA(t))
│   └── genie_refine.py          # Alias for LLaDAPipeline (swap for GENIE checkpoint)
│
├── encoders/
│   ├── base.py                  # Abstract TextEncoder interface
│   ├── clip_encoder.py          # CLIP-H/14 (77-token, hidden_dim=1024)
│   └── longclip_encoder.py      # Long-CLIP-L (248-token, hidden_dim=768)
│
├── rewriters/
│   ├── base.py                  # Abstract PromptRewriter interface
│   ├── ollama_rewriter.py       # AR baseline: Llama 3 via local Ollama server
│   └── llada_rewriter.py        # Diffusion rewriter: LLaDA-8B-Instruct
│
├── generation/
│   └── t2i_runner.py            # SD 2.1 wrapper; injects conditioning via prompt_embeds
│                                #   Includes optional EmbeddingProjector for LongCLIP→SD 2.1
│
├── evaluation/
│   ├── metrics.py               # CLIPScorer, FIDScorer, AttributeBindingScorer (BLIP-2 VQA),
│   │                            #   RelationAccuracyScorer (BLIP-2 VQA)
│   ├── embedding_analysis.py    # RQ1: semantic density, CLIP embedding cosine separation
│   ├── cfg_sensitivity.py       # RQ3: CFG sweep, CFGSweepResult, compositional stability
│   └── qualitative.py           # Rater annotation loader (CSV), Cohen's κ
│
├── experiments/
│   ├── run_experiment.py        # Main CLI entry point
│   ├── rq1_conditioning.py      # Semantic density + embedding separation analysis
│   ├── rq2_compositional.py     # Full generation + scoring benchmark
│   ├── rq3_cfg_sensitivity.py   # CFG sweep across all pipelines
│   └── rq4_mechanism.py         # AR vs LLaDA head-to-head with delta metrics
│
├── utils/
│   ├── seed.py                  # set_seed(), get_generator() for reproducibility
│   ├── logging.py               # W&B integration (init, log_metrics, log_images, finish)
│   ├── prompt_io.py             # load_prompts(), expansion ratio logging
│   └── truncation_monitor.py    # TruncationMonitor — per-pipeline truncation event tracking
│
├── scripts/
│   ├── prepare_prompts.sh       # Download and format T2I-CompBench++ and auxiliary sets
│   ├── submit_hpc.sh            # SLURM job templates
│   └── run_local.sh             # Single-GPU dry run for development
│
├── test_pipeline.py             # Smoke test (CPU-safe, no GPU required)
└── requirements.txt
```

---

## Setup

### Requirements

- Python **3.10–3.12** (3.14 is not yet supported by `tokenizers` / `safetensors`)
- CUDA-capable GPU — required for LLaDA (16 GB VRAM) and SD 2.1 (~5 GB VRAM)
- [Ollama](https://ollama.com) running locally for the AR rewrite condition

```bash
# 1. Create a virtual environment with Python 3.12
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install PyTorch with CUDA (pick your version)
#    CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#    CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#    CPU only (smoke tests / development):
pip install torch torchvision

# 3. Install all other dependencies
pip install -r requirements.txt
```

### Ollama (AR baseline)

```bash
ollama serve
```

`llama3.1` is pulled automatically on first use — no manual `ollama pull` required. The AR rewriter sends prompts to `http://localhost:11434` using the same expansion instruction as LLaDA, so the generative mechanism is the only variable in RQ4.

### LLaDA (diffusion rewriter)

LLaDA-8B-Instruct is downloaded automatically from Hugging Face on first use (~16 GB). No manual checkpoint setup required. Requires CUDA.

### Long-CLIP encoder

The encoder tries to download `longclip-L.pt` automatically from several candidate HuggingFace repos. If auto-download fails (the repo ID may have changed since this was written), download manually:

- HuggingFace: https://huggingface.co/BeichenZhang/LongCLIP-L
- GitHub: https://github.com/beichenzbc/Long-CLIP

Then set the path in config:

```yaml
longclip_checkpoint: /path/to/longclip-L.pt
```

The smoke test (`python test_pipeline.py`) skips Long-CLIP with a clear message if the checkpoint is unavailable, so you can verify everything else first.

---

## Running Experiments

### Smoke test (no GPU required)

```bash
python test_pipeline.py
```

Verifies CLIP and Long-CLIP encoding on CPU, reports Ollama and CUDA availability.

### Dry run (CPU-safe, fast)

```bash
python experiments/run_experiment.py --rq 1 --dry-run
```

Runs 3 prompts through all non-LLaDA pipelines with 2 denoising steps. Used to verify the full stack before committing GPU hours.

### Full runs

```bash
python experiments/run_experiment.py --rq 1   # Conditioning structure analysis (no image gen)
python experiments/run_experiment.py --rq 2   # Compositional benchmark (CLIPScore, FID, VQA)
python experiments/run_experiment.py --rq 3   # CFG sensitivity sweep
python experiments/run_experiment.py --rq 4   # AR vs LLaDA mechanism comparison
python experiments/run_experiment.py --rq all # All RQs sequentially
```

Override config fields on the CLI:

```bash
python experiments/run_experiment.py --rq 2 --seed 456 --cfg 5.0
python experiments/run_experiment.py --rq 3 --config sampling_steps=20
python experiments/run_experiment.py --rq 2 --no-wandb   # disable W&B
```

### HPC / SLURM

```bash
./scripts/submit_hpc.sh
```

Submits separate SLURM jobs for each RQ. See the script for partition and memory settings.

---

## Configuration

`config/experiment.yaml` controls all experimental parameters. Key fields:

```yaml
# Models
t2i_model: stabilityai/stable-diffusion-2-1
clip_model: laion/CLIP-ViT-H-14-laion2B-s32B-b79K
longclip_checkpoint: null          # null = auto-download from HF Hub
llada_model: GSAI-ML/LLaDA-8B-Instruct
ollama_model: llama3

# Generation
sampling_steps: 50
cfg_scales: [1.0, 3.0, 5.0, 7.5, 10.0]
default_cfg_scale: 7.5

# Experimental design
encoders: [clip, longclip]
pipelines: [raw, ar, llada]
seeds: [42, 123, 456, 789, 1024]

# Evaluation
vqa_model: Salesforce/blip2-flan-t5-xl
clip_score_model: openai/clip-vit-large-patch14
n_images_per_prompt: 5

# Tracking
wandb_project: prompt-pipeline
output_dir: outputs/
```

`config/prompts.yaml` contains real T2I-CompBench++ validation prompts (Huang et al.) plus two auxiliary sets:

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

| Metric | Implementation | Target RQ |
|--------|---------------|-----------|
| CLIPScore | CLIP cosine similarity (image vs prompt embedding) | RQ1, RQ2, RQ4 |
| FID | Fréchet Inception Distance vs reference set | RQ2, RQ4 |
| Attribute Binding Accuracy | BLIP-2 VQA: "Is there a `<attr>` `<obj>`?" | RQ2, RQ4 |
| Relation Accuracy | BLIP-2 VQA: "Is the `<A>` `<rel>` the `<B>`?" | RQ2, RQ4 |
| Compositional Stability | Variance of above metrics across CFG scales | RQ3 |
| Semantic Density | Attribute/relation token count in rewritten prompts | RQ1 |
| Embedding Separation | Mean pairwise cosine distance across prompt set | RQ1 |

Truncation events (tokens lost when rewritten prompts exceed the encoder window) are tracked separately by `TruncationMonitor` to distinguish encoder bottleneck effects from refinement quality effects.

Results are logged to Weights & Biases and saved as text summaries under `outputs/rqN/`.

---

## Key Design Decisions

**`prompt_embeds` injection.** The T2I runner bypasses SD 2.1's internal text encoder and passes conditioning tensors directly via diffusers' `prompt_embeds` parameter. This means the conditioning pathway is entirely under our control without modifying the backbone.

**LongCLIP → SD 2.1 dimension mismatch.** SD 2.1's UNet cross-attention expects (seq=77, dim=1024) from its CLIP-H encoder. Long-CLIP-L outputs (seq=248, dim=768). `T2IRunner` includes an optional `EmbeddingProjector` (linear dim projection + sequence interpolation) for this path. The projector starts with random weights and must be fine-tuned; for the main experiments, CLIP-H is used throughout and Long-CLIP serves as an ablation.

**Shared interface across conditioning paths.** All three pipelines implement `ConditioningPipeline.encode(prompt) → EncodingResult`, so `T2IRunner` and all evaluation modules are agnostic to which strategy or encoder they receive.

**Matched expansion instructions.** `OllamaRewriter` and `LLaDARewriter` use the exact same expansion instruction string. The only variable in RQ4 is the generative mechanism (left-to-right autoregressive vs masked diffusion).

**No fine-tuning of the backbone.** SD 2.1 weights are frozen throughout all experiments.

---

## Citation

If you build on this work, please cite the underlying proposal:

```
Nailana, O. (2026). Diffusion-Based Refinement of Textual Conditioning for Improved
Generative Modeling. MSc Research Proposal, University of the Witwatersrand.
Supervised by Dr. Devon Jarvis.
```

Key references: LLaDA (Nie et al., 2025), GENIE (Lin et al., 2023), T2I-CompBench++ (Huang et al., 2024), Long-CLIP (Zhang et al., 2024), Stable Diffusion (Rombach et al., 2022), BLIP-2 (Li et al., 2023).
