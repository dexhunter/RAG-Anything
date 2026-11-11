"""Baseline for ContextExtractor._extract_page_context.

During multimodal ingest every figure asks this method for a +/- 1 page
window. The implementation walks the whole content list on every call.
Workload: a 300-page illustrated manual, 12 text blocks per page and one
figure per page (300 figures), the same extractor instance ingest reuses.
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from raganything.modalprocessors import ContextConfig, ContextExtractor

_SENTENCE = "Paragraph text about the method and the reported results. "


def _content_list() -> tuple[list, list]:
    blocks = []
    figures = []
    for page in range(300):
        for block_index in range(12):
            blocks.append(
                {
                    "type": "text",
                    "text": _SENTENCE * 8,
                    "page_idx": page,
                    "text_level": 1 if block_index == 0 else 0,
                }
            )
        figure = {
            "type": "image",
            "img_path": f"images/figure_{page}.png",
            "page_idx": page,
        }
        blocks.append(figure)
        figures.append(figure)
    return blocks, figures


def _run(extractor: ContextExtractor, blocks: list, figures: list) -> int:
    total = 0
    for figure in figures:
        context = extractor._extract_page_context(
            blocks, {"page_idx": figure["page_idx"], "index": 0}
        )
        total += len(context)
    return total


def main() -> None:
    blocks, figures = _content_list()
    extractor = ContextExtractor(
        config=ContextConfig(
            context_window=1, context_mode="page", max_context_tokens=2000
        )
    )
    expected = _run(extractor, blocks, figures)
    samples = []
    for _ in range(5):
        gc.collect()
        started = time.perf_counter()
        got = _run(extractor, blocks, figures)
        samples.append((time.perf_counter() - started) * 1000.0)
        if got != expected:
            raise SystemExit(f"result changed: {got} != {expected}")
    median = statistics.median(samples)
    noise = (max(samples) - min(samples)) / median * 100.0 if median else 0.0
    print(
        json.dumps(
            {
                "function": "ContextExtractor._extract_page_context",
                "baseline_median_ms": median,
                "baseline_noise_pct": noise,
                "runs_ms": samples,
                "checksum": expected,
                "figures": len(figures),
                "blocks": len(blocks),
            }
        )
    )


if __name__ == "__main__":
    main()
