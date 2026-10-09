# Related work — prompt rewriting for T2I (and how our result fits)

Framing for §2 (related work) and the discussion. Our image-level finding — **raw
beats rewriting on compositional fidelity; PoE is the best rewriter but doesn't surpass
raw** — is consistent with a real and growing thread once "rewriting" is split into two
families that are often conflated.

## Family 1 — rewriting-as-expansion (denser prose)

This is where our `ar` (Llama) and `llada_single` conditions sit: expand the prompt into
longer, more descriptive text. Evidence is **mixed**.

- **DALL·E 3 — "Improving Image Generation with Better Captions" (Betker et al., 2023).**
  Descriptive captions dramatically improve prompt following. *Crucial caveat for us:*
  the gain comes mostly from **retraining** on synthetic descriptive captions; inference
  "prompt upsampling" rides on top. **Not** evidence that rewriting helps a *frozen*
  model like SDXL — pre-empt the reviewer who raises it.
  https://cdn.openai.com/papers/dall-e-3.pdf
- **Promptist — "Optimizing Prompts for Text-to-Image Generation" (Hao et al., 2022).**
  RL-tuned rewriter for Stable Diffusion; improves aesthetics / human preference. Targets
  quality, not strict compositional binding. https://arxiv.org/abs/2212.09611
- **PromptEnhancer — "Taming Your Rewriter via Fine-Grained Reward" (Wang et al., CVPR
  2026).** Most direct support for our result: its premise is that off-the-shelf
  rewriters **introduce inconsistencies and degrade fidelity**, needing reward alignment
  to the T2I model to be safe. "Naive rewriting can hurt" is now the problem statement,
  not a surprise.
  https://openaccess.thecvf.com/content/CVPR2026/html/Wang_PromptEnhancer_Taming_Your_Rewriter_for_Text-to-Image_Generation_via_Fine-Grained_Reward_CVPR_2026_paper.html

Why expansion underperforms on binding: a frozen text encoder (CLIP/T5, limited
capacity, weak compositional structure) doesn't reward denser text — added attributes
dilute and compete. Our raw>rewrite-on-binding result replicates this cleanly, with
significance, on a controlled compositional set.

## Family 2 — rewriting-as-planning / grounding (structure, not more words)

Reliably helps compositionality — but by changing the **conditioning structure**.

- **RPG — "Mastering Text-to-Image Diffusion: Recaptioning, Planning, Generating with
  MLLMs" (Yang et al., ICML 2024).** MLLM recaptions + plans sub-regions; large gains on
  attribute/spatial composition. https://proceedings.mlr.press/v235/yang24ai.html
- **LLM-grounded Diffusion (Lian et al., 2023).** LLM → layout (boxes) → grounded
  generation; gains on counting / spatial / attribute. (arXiv 2305.13655)

Lesson: what helps compositionality is not more words but stopping constraints from
competing in one dense text vector.

## Family 3 — composition at the model level (where PoE sits)

- **Composable Diffusion — "Compositional Visual Generation with Composable Diffusion
  Models" (Liu et al., ECCV 2022).** Compose concepts via AND/NOT at the score level
  (product/sum of distributions) — the Product-of-Experts ancestor; composing separate
  conditionings beats a single combined prompt for conjunctions. Our shared-scene PoE is
  the same intuition at the **text-conditioning** level.
  https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770426.pdf

## Rewriting-free alternatives (for completeness / baselines)

Inference-time attention / embedding methods fix binding without rewriting:
- **Attend-and-Excite** (Chefer et al., 2023) https://arxiv.org/abs/2301.13826
- **CONFORM** (Meral et al., CVPR 2024)
  https://openaccess.thecvf.com/content/CVPR2024/papers/Meral_CONFORM_Contrast_is_All_You_Need_for_High-Fidelity_Text-to-Image_Diffusion_CVPR_2024_paper.pdf
- **Enhanced text embeddings for attribute binding** (Zarei et al., 2024)
  https://arxiv.org/pdf/2406.07844

## How to position our contribution

- Our result = Family 1's known failure, cleanly demonstrated: on a frozen SDXL,
  expansion-style rewriting (single / AR) does **not** beat the raw prompt on
  compositional fidelity (cite PromptEnhancer for "why", DALL·E 3 for the retraining
  distinction).
- Our method = a move toward Families 2–3: PoE stops cramming constraints into one
  vector, which is exactly why it is the **best rewriter** (significant on CLIP, recovers
  single-rewrite's relation loss) even though rewriting-as-expansion doesn't beat raw.
  Cite Composable Diffusion as the score-level ancestor and RPG/LMD as the
  "structure helps" evidence.
- Honest caveat in the writeup: DALL·E 3 is the "but rewriting works" objection —
  answer it with *recaptioning-plus-retraining vs inference rewriting on a frozen model*.
