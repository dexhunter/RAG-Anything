#!/usr/bin/env python3
"""Time one ProcessorMixin._text_chunk_page_labels call.

After text insertion, page annotation chunks the sanitized document with
LightRAG's tokenizer and then replays that chunking to recover character
spans. Both steps encode the full document.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

_HARNESS = Path(__file__).resolve().parent
_REPO = _HARNESS.parents[1]
sys.path.insert(0, str(_HARNESS))
sys.path.insert(0, str(_REPO))

from bench_common import emit, time_call  # noqa: E402

logging.disable(logging.CRITICAL)

import tiktoken  # noqa: E402
from lightrag.operate import chunking_by_token_size  # noqa: E402
from lightrag.utils import Tokenizer, compute_mdhash_id  # noqa: E402

from raganything.processor import ProcessorMixin  # noqa: E402
from raganything.utils import (  # noqa: E402
    build_sanitized_page_map,
    separate_content_with_page_map,
)


PARAGRAPH = (
    "The experiment compares retrieval accuracy across tables, figures, "
    "and equations in the appendix. R&D remains stable. "
)


class _Docs:
    def __init__(self, content: str) -> None:
        self._content = content

    async def get_by_id(self, doc_id: str):
        return {"content": self._content}


def build_case(blocks: int = 5000):
    content = [
        {
            "type": "text",
            "text": f"Heading {index}. " + PARAGRAPH * 6,
            "page_idx": index // 10,
        }
        for index in range(blocks)
    ]
    text, _multimodal, intervals = separate_content_with_page_map(content)
    encoder = Tokenizer("cl100k_base", tiktoken.get_encoding("cl100k_base"))
    sanitized, _mapped = build_sanitized_page_map(text, intervals)
    chunks = chunking_by_token_size(encoder, sanitized, None, False, 100, 1200)
    recorded = {
        compute_mdhash_id(chunk["content"], prefix="chunk-") for chunk in chunks
    }
    lightrag = type("LightRAG", (), {})()
    lightrag.chunking_func = chunking_by_token_size
    lightrag.tokenizer = encoder
    lightrag.chunk_overlap_token_size = 100
    lightrag.chunk_token_size = 1200
    lightrag.full_docs = _Docs(sanitized)
    processor = ProcessorMixin()
    processor.lightrag = lightrag
    processor.logger = logging.getLogger("bench")
    return processor, text, intervals, recorded


def main() -> None:
    processor, text, intervals, recorded = build_case()

    def call():
        labels = asyncio.run(
            processor._text_chunk_page_labels(
                "doc-bench", text, intervals, None, False, recorded
            )
        )
        if len(labels) != len(recorded):
            raise SystemExit(f"annotation returned {len(labels)} of {len(recorded)}")
        return labels

    call()
    median, noise, samples = time_call(call, warmup=1)
    emit(
        "ProcessorMixin._text_chunk_page_labels",
        median,
        noise,
        samples,
        text_chars=len(text),
        chunks=len(recorded),
    )


if __name__ == "__main__":
    main()
