"""Peak-memory baseline for ProcessorMixin._generate_content_based_doc_id.

Insert builds one doc id from the full content list by joining every block
into a single signature and hashing it. Workload: a 500-page OCR content list,
about 2.2k characters of text per page, plus an image path every fifth page.
tracemalloc is started after the content list exists, so the peak is memory
allocated inside the call.
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import tracemalloc
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from raganything.processor import ProcessorMixin

_PAGE = "The multimodal retrieval pipeline stores text and figures. " * 40


def _content_list() -> list:
    blocks = []
    for page in range(500):
        blocks.append({"type": "text", "text": f"{_PAGE} page {page}"})
        if page % 5 == 0:
            blocks.append(
                {"type": "image", "img_path": f"/data/images/figure_{page}.png"}
            )
    return blocks


def _peak(method, blocks: list) -> tuple[int, str]:
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    doc_id = method(blocks)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, doc_id


def main() -> None:
    blocks = _content_list()
    method = ProcessorMixin()._generate_content_based_doc_id
    peaks = []
    doc_id = None
    for _ in range(5):
        peak, got = _peak(method, blocks)
        if doc_id is None:
            doc_id = got
        elif got != doc_id:
            raise SystemExit(f"doc id changed: {got} != {doc_id}")
        peaks.append(peak)
    median = statistics.median(peaks)
    noise = (max(peaks) - min(peaks)) / median * 100.0 if median else 0.0
    print(
        json.dumps(
            {
                "function": "ProcessorMixin._generate_content_based_doc_id",
                "metric_kind": "peak_memory_bytes",
                "baseline_value": median,
                "baseline_noise_pct": noise,
                "runs": peaks,
                "doc_id": doc_id,
            }
        )
    )


if __name__ == "__main__":
    main()
