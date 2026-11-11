"""Baseline for raganything.mineru_content._text_value.

MinerU content_list_v2 stores each paragraph as a list of spans. Ingest calls
convert_mineru_content_list_v2, which calls _text_value once per paragraph (and
on captions). This workload is a 180-page export: 6 paragraphs per page, about
60 word-level spans each, with an inline equation every 17 spans.
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

from raganything.mineru_content import _text_value

_WORDS = (
    "method",
    "uses",
    "a",
    "local",
    "feature",
    "of",
    "the",
    "model,",
    "and",
)


def _paragraph(seed: int) -> list:
    spans = []
    for index in range(60):
        if index % 17 == 0:
            spans.append({"type": "equation_inline", "content": "x^2"})
        else:
            spans.append(
                {"type": "text", "content": _WORDS[(index + seed) % len(_WORDS)]}
            )
    return spans


def _documents() -> list:
    return [
        _paragraph(page * 6 + paragraph)
        for page in range(180)
        for paragraph in range(6)
    ]


def _run(paragraphs: list) -> int:
    total = 0
    for spans in paragraphs:
        total += len(_text_value(spans))
    return total


def main() -> None:
    paragraphs = _documents()
    expected = _run(paragraphs)
    samples = []
    for _ in range(5):
        gc.collect()
        started = time.perf_counter()
        got = _run(paragraphs)
        samples.append((time.perf_counter() - started) * 1000.0)
        if got != expected:
            raise SystemExit(f"result changed: {got} != {expected}")
    median = statistics.median(samples)
    noise = (max(samples) - min(samples)) / median * 100.0 if median else 0.0
    print(
        json.dumps(
            {
                "function": "_text_value",
                "baseline_median_ms": median,
                "baseline_noise_pct": noise,
                "runs_ms": samples,
                "checksum": expected,
            }
        )
    )


if __name__ == "__main__":
    main()
