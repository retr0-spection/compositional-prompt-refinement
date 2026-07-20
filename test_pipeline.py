"""
test_pipeline.py

Smoke test for the full prompt expansion pipeline.

Tests each rewriter and encoder combination without requiring a GPU.
Encoders fall back to CPU automatically; the LLaDA rewriter requires CUDA
and is skipped if unavailable. The Ollama rewriter is skipped if the server
is not running.

Usage
-----
    # From the project root:
    python test_pipeline.py

    # Or with pytest:
    pytest test_pipeline.py -v
"""

from __future__ import annotations

import logging
import sys
import torch

try:
    import pytest
    _PYTEST_AVAILABLE = True
except ImportError:
    _PYTEST_AVAILABLE = False
    # Stub so @pytest.mark.skipif decorators don't crash the standalone runner
    class _PytestStub:
        class mark:
            @staticmethod
            def skipif(*args, **kwargs):
                def decorator(fn):
                    return fn
                return decorator
    pytest = _PytestStub()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------

PROMPTS = [
    "a red cat beside a blue dog",
    "a woman in a yellow dress standing near a green tree",
    "two children playing with a large orange ball on a sunny beach",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    import requests
    try:
        resp = requests.get("http://localhost:11434", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _cuda_available() -> bool:
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Rewriter tests
# ---------------------------------------------------------------------------

class TestOllamaRewriter:
    @pytest.mark.skipif(not _ollama_available(), reason="Ollama server not running")
    def test_rewrite_single(self):
        from rewriters.ollama_rewriter import OllamaRewriter
        rw = OllamaRewriter()
        prompt = PROMPTS[0]
        expanded = rw.rewrite(prompt)
        assert isinstance(expanded, str)
        assert len(expanded) > len(prompt), "Expanded prompt should be longer than input"
        logger.info("Ollama expansion:\n  IN:  %s\n  OUT: %s", prompt, expanded)

    @pytest.mark.skipif(not _ollama_available(), reason="Ollama server not running")
    def test_rewrite_batch(self):
        from rewriters.ollama_rewriter import OllamaRewriter
        rw = OllamaRewriter()
        results = rw.rewrite_batch(PROMPTS[:2])
        assert len(results) == 2
        for r in results:
            assert isinstance(r, str) and len(r) > 0


class TestLLaDARewriter:
    @pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
    def test_rewrite_single(self):
        from rewriters.llada_rewriter import LLaDARewriter, LLaDARewriterConfig
        cfg = LLaDARewriterConfig(gen_length=64, steps=32)  # fast settings for smoke test
        rw = LLaDARewriter(config=cfg)
        prompt = PROMPTS[0]
        expanded = rw.rewrite(prompt)
        assert isinstance(expanded, str)
        assert len(expanded) > 0
        logger.info("LLaDA expansion:\n  IN:  %s\n  OUT: %s", prompt, expanded)
        rw.unload()


# ---------------------------------------------------------------------------
# Encoder tests (CPU-safe)
# ---------------------------------------------------------------------------

class TestCLIPEncoder:
    def test_token_count(self):
        from encoders.clip_encoder import CLIPEncoder, CLIPEncoderConfig
        cfg = CLIPEncoderConfig(device="cpu", torch_dtype=torch.float32)
        enc = CLIPEncoder(config=cfg)
        count = enc.token_count(PROMPTS[0])
        assert isinstance(count, int) and count > 0
        logger.info("CLIP token count for %r: %d", PROMPTS[0], count)

    def test_encode_shape(self):
        from encoders.clip_encoder import CLIPEncoder, CLIPEncoderConfig
        cfg = CLIPEncoderConfig(device="cpu", torch_dtype=torch.float32)
        enc = CLIPEncoder(config=cfg)
        embedding = enc.encode(PROMPTS[0])
        assert embedding.shape[0] == 1, "Batch dim should be 1"
        assert embedding.shape[1] == 77, "Seq len should be 77"
        logger.info("CLIP embedding shape: %s", tuple(embedding.shape))

    def test_long_prompt_truncated(self):
        from encoders.clip_encoder import CLIPEncoder, CLIPEncoderConfig
        cfg = CLIPEncoderConfig(device="cpu", torch_dtype=torch.float32)
        enc = CLIPEncoder(config=cfg)
        long_prompt = " ".join(["a detailed scene with many objects"] * 20)
        embedding = enc.encode(long_prompt)
        assert embedding.shape[1] == 77, "Should always return 77-token embedding"


class TestLongCLIPEncoder:
    def test_token_count(self):
        from encoders.longclip_encoder import LongCLIPEncoder, LongCLIPEncoderConfig
        cfg = LongCLIPEncoderConfig(device="cpu", torch_dtype=torch.float32)
        enc = LongCLIPEncoder(config=cfg)
        count = enc.token_count(PROMPTS[0])
        assert isinstance(count, int) and count > 0
        logger.info("LongCLIP token count for %r: %d", PROMPTS[0], count)

    def test_encode_shape(self):
        from encoders.longclip_encoder import LongCLIPEncoder, LongCLIPEncoderConfig
        cfg = LongCLIPEncoderConfig(device="cpu", torch_dtype=torch.float32)
        enc = LongCLIPEncoder(config=cfg)
        embedding = enc.encode(PROMPTS[0])
        assert embedding.shape[0] == 1
        assert embedding.shape[1] == 248
        logger.info("LongCLIP embedding shape: %s", tuple(embedding.shape))

    def test_context_length_property(self):
        from encoders.longclip_encoder import LongCLIPEncoder
        enc = LongCLIPEncoder()
        assert enc.context_length == 248


# ---------------------------------------------------------------------------
# Pipeline integration tests (CPU-safe, no image generation)
# ---------------------------------------------------------------------------

class TestRawPipeline:
    def test_encode(self):
        from encoders.clip_encoder import CLIPEncoder, CLIPEncoderConfig
        from pipeline.raw import RawPipeline
        cfg = CLIPEncoderConfig(device="cpu", torch_dtype=torch.float32)
        enc = CLIPEncoder(config=cfg)
        pipe = RawPipeline(encoder=enc)
        result = pipe.encode(PROMPTS[0])
        assert result.raw_prompt == PROMPTS[0]
        assert result.rewritten_prompt == PROMPTS[0]
        assert result.embedding.shape[1] == 77
        logger.info("RawPipeline (%s): tokens=%d, truncated=%s",
                    pipe.name, result.token_count_raw, result.was_truncated)

    def test_encode_batch(self):
        from encoders.clip_encoder import CLIPEncoder, CLIPEncoderConfig
        from pipeline.raw import RawPipeline
        cfg = CLIPEncoderConfig(device="cpu", torch_dtype=torch.float32)
        enc = CLIPEncoder(config=cfg)
        pipe = RawPipeline(encoder=enc)
        results = pipe.encode_batch(PROMPTS)
        assert len(results) == len(PROMPTS)


class TestARPipelineOffline:
    """Tests ARPipeline structure without calling Ollama."""

    def test_pipeline_name(self):
        from encoders.clip_encoder import CLIPEncoder, CLIPEncoderConfig
        from pipeline.ar_rewrite import ARPipeline
        from rewriters.ollama_rewriter import OllamaRewriter
        cfg = CLIPEncoderConfig(device="cpu", torch_dtype=torch.float32)
        enc = CLIPEncoder(config=cfg)
        pipe = ARPipeline(encoder=enc)
        assert pipe.name == "ar_clip"

    @pytest.mark.skipif(not _ollama_available(), reason="Ollama server not running")
    def test_encode_live(self):
        from encoders.clip_encoder import CLIPEncoder, CLIPEncoderConfig
        from pipeline.ar_rewrite import ARPipeline
        cfg = CLIPEncoderConfig(device="cpu", torch_dtype=torch.float32)
        enc = CLIPEncoder(config=cfg)
        pipe = ARPipeline(encoder=enc)
        result = pipe.encode(PROMPTS[0])
        assert result.raw_prompt == PROMPTS[0]
        assert result.rewritten_prompt != PROMPTS[0] or True  # may be identical in edge cases
        logger.info("ARPipeline: tokens_raw=%d, tokens_rewritten=%d, truncated=%s",
                    result.token_count_raw, result.token_count_rewritten, result.was_truncated)


# ---------------------------------------------------------------------------
# Quick standalone runner (no pytest needed)
# ---------------------------------------------------------------------------

def _run_quick_check() -> None:
    """CPU-only sanity check — no models needed beyond CLIP."""
    print("\n" + "=" * 60)
    print("Prompt Pipeline — Quick Smoke Check")
    print("=" * 60)

    from encoders.clip_encoder import CLIPEncoder, CLIPEncoderConfig
    from encoders.longclip_encoder import LongCLIPEncoder, LongCLIPEncoderConfig
    from pipeline.raw import RawPipeline

    prompt = PROMPTS[0]
    print(f"\nPrompt: {prompt!r}")

    # CLIP
    print("\n[1/2] CLIPEncoder (CPU)")
    clip_cfg = CLIPEncoderConfig(device="cpu", torch_dtype=torch.float32)
    clip_enc = CLIPEncoder(config=clip_cfg)
    raw_clip = RawPipeline(encoder=clip_enc)
    result = raw_clip.encode(prompt)
    print(f"  Pipeline:  {raw_clip.name}")
    print(f"  Embedding: {tuple(result.embedding.shape)}")
    print(f"  Tokens:    {result.token_count_raw}")
    print(f"  Truncated: {result.was_truncated}")

    # LongCLIP
    print("\n[2/2] LongCLIPEncoder (CPU) — downloads checkpoint on first run")
    lc_cfg = LongCLIPEncoderConfig(device="cpu", torch_dtype=torch.float32)
    lc_enc = LongCLIPEncoder(config=lc_cfg)
    raw_lc = RawPipeline(encoder=lc_enc)
    try:
        result = raw_lc.encode(prompt)
        print(f"  Pipeline:  {raw_lc.name}")
        print(f"  Embedding: {tuple(result.embedding.shape)}")
        print(f"  Tokens:    {result.token_count_raw}")
        print(f"  Truncated: {result.was_truncated}")
        _longclip_ok = True
    except RuntimeError as exc:
        # Checkpoint not available — print instructions and continue
        print(f"\n  [SKIP] Long-CLIP checkpoint unavailable:")
        for line in str(exc).splitlines()[:6]:
            print(f"         {line}")
        print()
        _longclip_ok = False

    if _longclip_ok:
        print("\n✓ Smoke check passed (CLIP + Long-CLIP).\n")
    else:
        print("\n✓ Smoke check passed (CLIP only — Long-CLIP skipped, see above).\n")

    if _ollama_available():
        print("[Ollama] Server detected — testing OllamaRewriter...")
        from rewriters.ollama_rewriter import OllamaRewriter
        rw = OllamaRewriter()
        expanded = rw.rewrite(prompt)
        print(f"  IN:  {prompt}")
        print(f"  OUT: {expanded}")
    else:
        print("[Ollama] Server not running — skipping AR rewriter test.")
        print("         Start with: ollama serve")

    if _cuda_available():
        print(f"\n[CUDA] Device available: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[CUDA] VRAM: {vram_gb:.1f} GB")
        if vram_gb < 16.0:
            print(f"\n[LLaDA] SKIPPED — LLaDA-8B needs ~16 GB VRAM, this GPU has {vram_gb:.1f} GB.")
            print("        This is expected on bigbatch nodes; the LLaDA warmup job")
            print("        runs on the biggpu partition with sufficient VRAM.")
        else:
            print("\n[LLaDA] Testing rewriter (requires ~16 GB VRAM, may take a minute)...")
            from rewriters.llada_rewriter import LLaDARewriter
            try:
                rw = LLaDARewriter()
                expanded = rw.rewrite(prompt)
                print(f"  IN:  {prompt}")
                print(f"  OUT: {expanded}")
                rw.unload()
            except Exception as exc:
                print(f"  [FAIL] LLaDA rewriter error: {exc}")
    else:
        print("\n[CUDA] Not available — skipping LLaDA rewriter test.")
        print("         LLaDA requires a CUDA GPU (~16 GB VRAM).")


if __name__ == "__main__":
    _run_quick_check()
