#!/usr/bin/env python3
"""Time one ContextExtractor._truncate_context call on a long single page.

Document text that shares page_idx 0 (a markdown export, or a content list
inserted without per-page breaks) is joined and then tokenized so the
extractor can keep max_context_tokens. The call encodes the whole page.
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

from lightrag.utils import Tokenizer  # noqa: E402
import tiktoken  # noqa: E402

from raganything.modalprocessors import ContextConfig, ContextExtractor  # noqa: E402


PARAGRAPH = (
    "The experiment compares retrieval accuracy across tables, figures, "
    "and equations in the appendix. R&D remains stable. "
)


def build_context(blocks: int = 5000) -> str:
    parts = [f"Heading {index}. " + PARAGRAPH * 6 for index in range(blocks)]
    # Same join as ContextExtractor._extract_page_context.
    return "\n".join(parts)


def main() -> None:
    context = build_context()
    encoder = Tokenizer("cl100k_base", tiktoken.get_encoding("cl100k_base"))
    extractor = ContextExtractor(
        ContextConfig(context_window=1, context_mode="page", max_context_tokens=2000),
        tokenizer=encoder,
    )
    kept = extractor._truncate_context(context)
    if len(kept) >= len(context):
        raise SystemExit(f"truncation did not shrink context: {len(kept)}")

    def call():
        return extractor._truncate_context(context)

    median, noise, samples = time_call(call)
    emit(
        "ContextExtractor._truncate_context",
        median,
        noise,
        samples,
        context_chars=len(context),
        kept_chars=len(kept),
    )


if __name__ == "__main__":
    main()
