"""
utils/truncation_monitor.py

Records CLIP truncation events per prompt per pipeline.

Standard CLIP's effective context is ~20 tokens despite a nominal limit of 77
(Zhang et al., 2024). Truncation monitor tracks how many tokens survive encoding
so null results in RQ4 can be attributed to encoder capacity rather than
refinement quality.

Usage
-----
monitor = TruncationMonitor()
monitor.record(pipeline_name="llada_clip", prompt="...", token_count_raw=12,
               token_count_rewritten=94, context_limit=77, was_truncated=True)
monitor.summary()          # prints per-pipeline truncation stats
monitor.to_dict()          # returns serialisable dict for W&B logging
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TruncationEvent:
    pipeline_name: str
    prompt: str
    token_count_raw: int
    token_count_rewritten: int
    context_limit: int
    was_truncated: bool

    @property
    def tokens_lost(self) -> int:
        return max(0, self.token_count_rewritten - self.context_limit)

    @property
    def expansion_ratio(self) -> float:
        if self.token_count_raw == 0:
            return 0.0
        return self.token_count_rewritten / self.token_count_raw


class TruncationMonitor:
    """
    Collects and summarises truncation events across all pipelines.

    One instance per experiment run. Pass the same instance to all pipelines
    or record events manually after calling pipeline.encode().
    """

    def __init__(self) -> None:
        self._events: list[TruncationEvent] = []

    def record(
        self,
        pipeline_name: str,
        prompt: str,
        token_count_raw: int,
        token_count_rewritten: int,
        context_limit: int,
        was_truncated: bool,
    ) -> None:
        evt = TruncationEvent(
            pipeline_name=pipeline_name,
            prompt=prompt,
            token_count_raw=token_count_raw,
            token_count_rewritten=token_count_rewritten,
            context_limit=context_limit,
            was_truncated=was_truncated,
        )
        self._events.append(evt)
        if was_truncated:
            logger.warning(
                "[%s] Truncated: %d -> %d tokens (limit %d, lost %d). Prompt: %r",
                pipeline_name, token_count_raw, token_count_rewritten,
                context_limit, evt.tokens_lost, prompt[:60],
            )

    def record_from_result(self, pipeline_name: str, result) -> None:
        """Convenience: record directly from an EncodingResult."""
        self.record(
            pipeline_name=pipeline_name,
            prompt=result.raw_prompt,
            token_count_raw=result.token_count_raw,
            token_count_rewritten=result.token_count_rewritten,
            context_limit=result.token_count_rewritten,  # stored in result
            was_truncated=result.was_truncated,
        )

    def summary(self) -> None:
        """Print per-pipeline truncation statistics to stdout."""
        by_pipeline: dict[str, list[TruncationEvent]] = defaultdict(list)
        for evt in self._events:
            by_pipeline[evt.pipeline_name].append(evt)

        print("\n=== Truncation Monitor Summary ===")
        for name, events in sorted(by_pipeline.items()):
            n_total = len(events)
            n_truncated = sum(1 for e in events if e.was_truncated)
            avg_expansion = sum(e.expansion_ratio for e in events) / n_total
            avg_lost = sum(e.tokens_lost for e in events) / n_total
            print(
                f"  {name}: {n_truncated}/{n_total} truncated "
                f"(avg expansion {avg_expansion:.2f}x, avg tokens lost {avg_lost:.1f})"
            )
        print()

    def to_dict(self) -> dict:
        """Return per-pipeline stats as a flat dict for W&B logging."""
        by_pipeline: dict[str, list[TruncationEvent]] = defaultdict(list)
        for evt in self._events:
            by_pipeline[evt.pipeline_name].append(evt)

        out = {}
        for name, events in by_pipeline.items():
            n_total = len(events)
            n_truncated = sum(1 for e in events if e.was_truncated)
            out[f"{name}/truncation_rate"] = n_truncated / n_total if n_total else 0.0
            out[f"{name}/avg_expansion_ratio"] = (
                sum(e.expansion_ratio for e in events) / n_total if n_total else 0.0
            )
            out[f"{name}/avg_tokens_lost"] = (
                sum(e.tokens_lost for e in events) / n_total if n_total else 0.0
            )
        return out

    @property
    def events(self) -> list[TruncationEvent]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)
