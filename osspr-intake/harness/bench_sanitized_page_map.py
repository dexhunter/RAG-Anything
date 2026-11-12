#!/usr/bin/env python3
"""Time one build_sanitized_page_map call on a multi-thousand-block document.

Page annotation sanitizes the joined text and every block, then locates
each block in the sanitized document.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_HARNESS = Path(__file__).resolve().parent
_REPO = _HARNESS.parents[1]
sys.path.insert(0, str(_HARNESS))
sys.path.insert(0, str(_REPO))

from bench_common import emit, time_call  # noqa: E402

logging.disable(logging.CRITICAL)

from raganything.utils import (  # noqa: E402
    build_sanitized_page_map,
    separate_content_with_page_map,
)


PARAGRAPH = (
    "The experiment compares retrieval accuracy across tables, figures, "
    "and equations in the appendix. R&D remains stable. "
)


def build_inputs(blocks: int = 5000):
    content = [
        {
            "type": "text",
            "text": f"Heading {index}. " + PARAGRAPH * 6,
            "page_idx": index // 10,
            "text_level": 1 if index % 15 == 0 else 0,
        }
        for index in range(blocks)
    ]
    text, _multimodal, intervals = separate_content_with_page_map(content)
    return text, intervals


def main() -> None:
    text, intervals = build_inputs()
    sanitized, mapped = build_sanitized_page_map(text, intervals)
    if len(mapped) != len(intervals) or not sanitized:
        raise SystemExit(
            f"unexpected map: intervals={len(intervals)} mapped={len(mapped)}"
        )

    def call():
        return build_sanitized_page_map(text, intervals)

    median, noise, samples = time_call(call)
    emit(
        "build_sanitized_page_map",
        median,
        noise,
        samples,
        text_chars=len(text),
        blocks=len(intervals),
    )


if __name__ == "__main__":
    main()
