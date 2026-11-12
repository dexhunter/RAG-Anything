"""Shared timing helpers for intake benchmarks.

No network and no model calls. The cl100k vocabulary is read from the
tiktoken cache shipped next to these scripts.
"""

from __future__ import annotations

import gc
import json
import os
import random
import statistics
import time
from pathlib import Path

# Must be set before tiktoken (or lightrag) resolves an encoding.
_CACHE = Path(__file__).resolve().parent / "encoding_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = str(_CACHE)

random.seed(0)


def time_call(fn, *, runs: int = 5, warmup: int = 2):
    """Return (median_ms, noise_pct, samples_ms).

    Noise is (max - min) / median * 100 over ``runs`` timed calls.
    ``gc.collect()`` runs before each sample so a collection does not
    land inside one of them.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        gc.collect()
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    median = statistics.median(samples)
    noise = (max(samples) - min(samples)) / median * 100.0 if median else 0.0
    return median, noise, samples


def emit(function: str, median_ms: float, noise_pct: float, samples, **extra) -> None:
    payload = {
        "function": function,
        "median_ms": round(median_ms, 3),
        "noise_pct": round(noise_pct, 3),
        "runs_ms": [round(sample, 3) for sample in samples],
        "runs": len(samples),
        **extra,
    }
    print(json.dumps(payload, sort_keys=True))
