#!/usr/bin/env python3
"""Time one MineruParser._read_output_files call on a large content list.

MinerU writes one relative image path per figure. The reader turns each
path into an absolute path before ingestion continues.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

_HARNESS = Path(__file__).resolve().parent
_REPO = _HARNESS.parents[1]
sys.path.insert(0, str(_HARNESS))
sys.path.insert(0, str(_REPO))

from bench_common import emit, time_call  # noqa: E402

logging.disable(logging.CRITICAL)

from raganything.parser import MineruParser  # noqa: E402


def build_output(root: Path, images: int = 3000) -> None:
    parse_dir = root / "paper" / "auto"
    image_dir = parse_dir / "images"
    image_dir.mkdir(parents=True)
    items = []
    for index in range(images):
        name = f"fig_{index}.jpg"
        (image_dir / name).write_bytes(b"\xff\xd8\xff")
        items.append(
            {
                "type": "image",
                "img_path": f"images/{name}",
                "image_caption": [f"Figure {index} caption"],
                "page_idx": index % 40,
            }
        )
    (parse_dir / "paper_content_list.json").write_text(json.dumps(items))
    (parse_dir / "paper.md").write_text("parsed markdown")


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="rag-intake-mineru-"))
    try:
        build_output(root)
        content, _markdown = MineruParser._read_output_files(root, "paper", "auto")
        if len(content) != 3000 or not content[0]["img_path"].endswith("fig_0.jpg"):
            raise SystemExit(f"unexpected read result: {len(content)}")

        def call():
            return MineruParser._read_output_files(root, "paper", "auto")

        median, noise, samples = time_call(call)
        emit(
            "MineruParser._read_output_files",
            median,
            noise,
            samples,
            images=3000,
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
