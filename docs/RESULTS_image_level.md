# RQ5 â€” Image-level results (SDXL capstone)

**Run:** SLURM job 65826, backbone SDXL, 100 compositional prompts Ã— 4 conditions,
10h18m, completed 2026-10-09. Images in `outputs/sdxl/rq5/<condition>/`, metrics in
`outputs/sdxl/rq5/rq5_{aggregate,comparison,per_type}.json` and `rq5_summary.txt`.
Relation accuracy re-scored with the fixed parser (job 65991, `relation_rescore.jsonl`).

Conditions: `raw` (original prompt, no rewrite), `ar` (Llama-3.1 rewrite via Ollama),
`llada_single` (one whole-prompt LLaDA rewrite), `llada_poe` (shared-scene
Product-of-Experts). All scored **against the original prompt** (user intent).

## Headline numbers (n = 100)

| Condition | Attr binding | Relation accâ€  | CLIPScore |
|-----------|:------------:|:-------------:|:---------:|
| **raw**         | **0.948** | **0.490** | **0.302** |
| ar              | 0.908 | 0.430 | 0.277 |
| llada_single    | 0.907 | 0.450 | 0.283 |
| **llada_poe**   | 0.915 | 0.470 | 0.291 |

â€ Relation accuracy after the parser fix (see below). Pre-fix values were raw 0.425 /
ar 0.395 / single 0.315 / poe 0.385 â€” all depressed by the bug, single the most.

## Paired significance (PoE vs each)

- **CLIP (Wilcoxon):** raw > PoE (Î” âˆ’0.011, **p = 0.001**); PoE > AR (Î” +0.014,
  **p = 0.001**); PoE > single (Î” +0.008, **p = 0.045**).
- **Attr binding (Wilcoxon):** all differences NS (p â‰¥ 0.1); conditions â‰ˆ 0.91â€“0.95.
- **Relation acc (sign test, fixed parser):** PoE vs raw Î” âˆ’0.020 (p = 0.79);
  PoE vs AR Î” +0.040 (p = 0.56); PoE vs single Î” +0.020 (p = 0.84) â€” **all NS**.

## The verdict

1. **No rewriting beats the raw prompt.** Raw is highest on all three metrics and
   significantly higher CLIP than every rewrite. At the image level â€” consistent with
   the text stage â€” refinement does **not** improve compositional fidelity over the
   original prompt for these prompts on SDXL.
2. **Among rewriting methods, PoE is the best â€” but the margin is CLIP-driven.** PoE
   significantly beats single and AR on **CLIP**. On attribute binding and relations the
   rewriters are statistically indistinguishable (all NS). PoE edges single/AR on
   relations by a hair (0.470 vs 0.450/0.430) but it is **not** the large "PoE recovers
   the relations single throws away" effect the buggy scorer suggested (that apparent
   +0.070 gap shrank to +0.020, NS, once questions were well-formed).
3. **Binding/relation gaps are within noise at n = 100** â€” only CLIP separates the
   methods significantly.

**Thesis framing:** PoE is the principled way to *do* refinement (no degeneracy, best
CLIP of any rewriter, never worse than the other rewriters), but refinement itself
doesn't beat the raw prompt. Do **not** overclaim a relation-specific advantage â€”
state it as "PoE is no worse, and best on CLIP." Matches RQ2.

## Relation-scorer bug (found, fixed, re-scored â€” RESOLVED)

`evaluation.metrics.parse_spatial_relations` took the relation **object** as the
*first*&››Û‹X\XÛHÛÜ™Y\ˆH™[][Ûˆ8 %H
Š˜Y™Xİ]™JŠˆ
˜›YHŠK›İH›İ[‚Š™ÙÈŠH8 %Ú[HHİXš™XİÛÜœ™XİHÛÚÈH›İ[‹ˆ]™\H™[][Ûˆ”PH]Y\İ[ÛˆØ\Â›X[›Ü›YY
’\ÈHØ]ÛˆHYÙˆH
˜›YJÈŠK\™\ÜÚ[™È™[][ÛˆXØİ\˜XŞH›Ü‚˜[›İ\ˆÛÛ™][ÛœÈ
˜[šÚ[™È›İYÚH[XœÛÛ]H[X™\œÈ[™\œİ]Y
Kˆ]šX]B˜š[™[™È[™ÓT[˜Y™™XİY‚‚ŠŠ‘š^ŠŠˆØš™XİHXY›İ[ˆÙˆ]È˜\ÙH
\İ›Û‹X\XÛHÛÜ™
K›İ[™YHH™^œÜ]X[™[][ÛˆÈÛÛ[XHÛÈÚZ[™Y›Û\È
¸ )˜HÜ™Y[ˆœ›ÙÈ[ˆHœ›İÛˆÛ™ŠHZÙBˆ™œ›ÙÈ‹›İœÛ™‹ˆ™\šYšYYÛˆ[L›Û\ÎˆNKÌLH™[][ÛœÈ›İÈ\œÙHÛÜœ™XİK‚‚ŠŠ•Ú]H™K\ØÛÜ™HÚ[™ÙYŠŠˆ[™[][Ûˆ[X™\œÈ›ÜÙH
YÈ\™\ÜÙY]™\][™ÊNÂŠŠœÚ[™ÛK\™]Üš]H›ÜÙH[Üİ

ÌŒLÍJJŠ‹™]™X[[™ÈHYÙŞHØÛÜ™\ˆY[™˜Z\›H[˜[\ÙYœÚ[™ÛIÜÈ™\˜›ÜÙH™]Üš]\ËˆÑIÜÈ™[][ÛˆXYİ™\ˆÚ[™ÛHÛÛ\ÙYœ›ÛH
ÌŒÌÂŠÌŒŒ
”ÊKˆHXY[™H
˜]È™\İÈÑH™\İ™]Üš]\ˆšXHÓT
H\È[˜Ú[™ÙY8 %]Bœ™[][ÛˆİÜH\È›İÈ”ÑH\È›ÈÛÜœÙH[ˆÚ[™ÛH‹›İ”ÑH™XÛİ™\œÈ™[][ÛœÈ‹‚Ø]™X]›ÜˆHÜš]]\ˆ™[][ÛˆXØİ\˜XŞH\Èİ[Û›HŒø $ÌKÛÈÜ]X[™[][ÛœÂ˜\™H\™›Üˆ[ÛÛ™][ÛœÈÛˆÑÈ™X]™[][Ûˆ[X™\œÈ\È\™Xİ[Û˜[