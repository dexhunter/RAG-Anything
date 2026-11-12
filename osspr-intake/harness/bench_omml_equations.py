#!/usr/bin/env python3
"""Time one extract_omml_equations call on a DOCX with thousands of equations.

The reader parses word/document.xml, converts each top-level m:oMath
element to LaTeX, and also serializes that element back to XML.
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

_HARNESS = Path(__file__).resolve().parent
_REPO = _HARNESS.parents[1]
sys.path.insert(0, str(_HARNESS))
sys.path.insert(0, str(_REPO))

from bench_common import emit, time_call  # noqa: E402

logging.disable(logging.CRITICAL)

from raganything.omml_extractor import extract_omml_equations  # noqa: E402

_MATH = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_WORD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def build_docx(path: Path, equations: int = 4000) -> None:
    parts = []
    for index in range(equations):
        parts.append(
            "<w:p><m:oMath>"
            f"<m:f><m:num><m:r><m:t>a{index}+x</m:t></m:r></m:num>"
            f"<m:den><m:r><m:t>b{index}+y</m:t></m:r></m:den></m:f>"
            "</m:oMath></w:p>"
        )
    document = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<w:document xmlns:w="{_WORD}" xmlns:m="{_MATH}"><w:body>'
        + "".join(parts)
        + "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"></Types>',
        )
        archive.writestr("word/document.xml", document)


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="rag-intake-omml-"))
    docx = root / "paper.docx"
    try:
        build_docx(docx)
        extracted = extract_omml_equations(docx)
        if len(extracted) != 4000 or extracted[0]["text"] != r"\frac{a0+x}{b0+y}":
            raise SystemExit(f"unexpected extraction: {extracted[:1]}")

        def call():
            return extract_omml_equations(docx)

        median, noise, samples = time_call(call)
        emit(
            "extract_omml_equations",
            median,
            noise,
            samples,
            equations=4000,
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
