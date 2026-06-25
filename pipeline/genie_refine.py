"""
pipeline/genie_refine.py

Alias: LLaDAPipeline is the diffusion-based refinement pipeline used in
this implementation. The proposal refers to GENIE (Lin et al., 2023);
LLaDA-8B-Instruct is used here as a publicly available masked diffusion LM
with the same generative mechanism (iterative masked token denoising).

If the original GENIE checkpoint becomes available, substitute the rewriter:

    from rewriters.genie_rewriter import GENIERewriter
    pipeline = GENIEPipeline(encoder=enc, rewriter=GENIERewriter())

For now, GENIEPipeline is an alias for LLaDAPipeline.
"""

from pipeline.llada_refine import LLaDAPipeline as GENIEPipeline

__all__ = ["GENIEPipeline"]
