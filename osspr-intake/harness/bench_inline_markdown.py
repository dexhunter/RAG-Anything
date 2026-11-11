"""Baseline for Parser._process_inline_markdown.

Parser.convert_text_to_pdf walks a markdown file line by line and sends every
non-empty line through _process_inline_markdown, which recompiles its
expressions on each call. Workload: a 250-page technical markdown manual,
32 lines per page, mixing headings, bold, italics, code, links, and strike.
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

from raganything.parser import Parser


def _lines() -> list[str]:
    lines = []
    for index in range(250 * 32):
        if index % 32 == 0:
            lines.append(
                f"## Section {index} with **bold terms** and a "
                f"[link](https://example.com/spec/{index})"
            )
        elif index % 7 == 0:
            lines.append(
                f"Use `code_{index}` and *italic term* plus score < {index} and __strong__."
            )
        else:
            lines.append(
                f"The method uses feature {index} of the local model and compares ~~old~~ values."
            )
    return lines


def _run(lines: list[str]) -> int:
    total = 0
    render = Parser._process_inline_markdown
    for line in lines:
        total += len(render(line))
    return total


def main() -> None:
    lines = _lines()
    expected = _run(lines)
    samples = []
    for _ in range(5):
        gc.collect()
        started = time.perf_counter()
        got = _run(lines)
        samples.append((time.perf_counter() - started) * 1000.0)
        if got != expected:
            raise SystemExit(f"result changed: {got} != {expected}")
    median = statistics.median(samples)
    noise = (max(samples) - min(samples)) / median * 100.0 if median else 0.0
    print(
        json.dumps(
            {
                "function": "Parser._process_inline_markdown",
                "baseline_median_ms": median,
                "baseline_noise_pct": noise,
                "runs_ms": samples,
                "checksum": expected,
                "lines": len(lines),
            }
        )
    )


if __name__ == "__main__":
    main()
