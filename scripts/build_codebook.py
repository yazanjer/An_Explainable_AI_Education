#!/usr/bin/env python3
"""Extract the official PISA 2018 student-questionnaire codebook from the
OECD instrument PDF into a versioned YAML file.

Answers editor comment 7: "Provide official PISA item descriptions and coding
for every interpreted variable. Do not assign educational meanings to coded
features without documentary support."

The original manuscript glossed items from memory and got them wrong (see
CHANGELOG_REVISION.md entry CB-1). Everything downstream that names a variable
must resolve it through this file.

Source document
---------------
    CY7_201709_QST_MS_STQ_CBA_NoNotes
    "PISA 2018 Student Questionnaire (Main Survey, computer-based)"
    OECD, https://www.oecd.org/en/data/datasets/pisa-2018-database.html

Usage
-----
    python scripts/build_codebook.py --pdf <path to questionnaire pdf> \
        --out config/codebook_pisa2018_student.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Dict, List

import yaml

ITEM_RE = re.compile(r"\b(ST\d{3}[A-Z]?Q\d{2}[A-Z]{2})\b")
BLOCK_LINE_RE = re.compile(r"^(ST\d{3}[A-Z]?)$")
BLOCK_ANY_RE = re.compile(r"\b(ST\d{3}[A-Z]?)\b(?!Q)")
HEADER_RE = re.compile(r"^CY7_\d+_QST")
INSTR_RE = re.compile(r"^\(Please\b|^\(Move\b|^\(Select\b", re.I)
FILTER_RE = re.compile(r"^\[")
CODES_RE = re.compile(r"^(?:\d{1,2}\s+){1,9}\d{1,2}$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_page(text: str, page_no: int) -> List[Dict]:
    """Parse one questionnaire page into zero or more item blocks."""
    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]
    lines = [ln for ln in lines if not HEADER_RE.match(ln)]
    if lines and lines[-1].strip().isdigit():
        lines = lines[:-1]  # trailing page number

    item_codes: List[str] = []
    for ln in lines:
        for m in ITEM_RE.finditer(ln):
            if m.group(1) not in item_codes:
                item_codes.append(m.group(1))
    if not item_codes:
        return []

    block = None
    block_idx = None
    for i, ln in enumerate(lines):
        if BLOCK_LINE_RE.match(ln.strip()):
            block, block_idx = ln.strip(), i
            break
    if block is None:
        block = re.match(r"(ST\d{3}[A-Z]?)Q", item_codes[0]).group(1)
        block_idx = len(lines)

    stem = " ".join(
        ln for ln in lines[:block_idx]
        if not INSTR_RE.match(ln) and not ITEM_RE.search(ln)
    ).strip()

    instruction = next((ln for ln in lines if INSTR_RE.match(ln)), "")
    filter_note = " ".join(ln for ln in lines if FILTER_RE.match(ln)).strip()

    # Response-scale header: non-item, non-instruction lines that sit between
    # the instruction and the first line carrying an item code.
    first_item_line = next(
        (i for i, ln in enumerate(lines) if ITEM_RE.search(ln)), len(lines)
    )
    scale_lines = [
        ln for ln in lines[block_idx + 1: first_item_line]
        if not INSTR_RE.match(ln) and not FILTER_RE.match(ln)
    ]
    scale = " ".join(scale_lines).strip()

    records = []
    for code in item_codes:
        label_parts = []
        codes_seen: List[str] = []
        for i, ln in enumerate(lines):
            if code in ln:
                rest = ITEM_RE.sub("", ln).strip()
                # Strip any other item code that shares the line (ST019 layout)
                rest = re.sub(r"\s{2,}", " ", rest).strip()
                if rest and not rest[0].isdigit():
                    label_parts.append(rest)
                elif rest:
                    codes_seen.append(rest)
                # A label may wrap onto the preceding or following line.
                for j in (i - 1, i + 1):
                    if 0 <= j < len(lines):
                        nb = lines[j]
                        if ITEM_RE.search(nb) or INSTR_RE.match(nb):
                            continue
                        if CODES_RE.match(nb.strip()):
                            codes_seen.append(nb.strip())
                        elif not label_parts and j == i + 1:
                            label_parts.append(nb.strip())
        label = " ".join(dict.fromkeys(p for p in label_parts if p)).strip()
        label = re.sub(r"\s+", " ", label)
        records.append(
            {
                "item": code,
                "block": block,
                "page_pdf": page_no + 1,
                "question_stem": stem,
                "item_label": label,
                "response_scale": scale,
                "response_codes_raw": " | ".join(dict.fromkeys(codes_seen)),
                "instruction": instruction,
                "filter_note": filter_note,
                "needs_manual_review": not bool(label) or not bool(scale),
            }
        )
    return records


def build(pdf_path: Path) -> Dict:
    import pdfplumber

    entries: Dict[str, Dict] = {}
    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            for rec in parse_page(page.extract_text() or "", i):
                entries.setdefault(rec["item"], rec)

    return {
        "codebook_version": "1.0",
        "generated": date.today().isoformat(),
        "generator": "scripts/build_codebook.py",
        "source": {
            "document": "PISA 2018 Student Questionnaire (Main Survey, computer-based)",
            "internal_id": "CY7_201709_QST_MS_STQ_CBA_NoNotes",
            "publisher": "OECD",
            "url": "https://www.oecd.org/en/data/datasets/pisa-2018-database.html",
            "file_name": pdf_path.name,
            "file_sha256": sha256(pdf_path),
            "n_pages": n_pages,
        },
        "note": (
            "Auto-extracted from the official instrument. Items flagged "
            "needs_manual_review have an incomplete label or scale and MUST be "
            "verified against the PDF before being interpreted in the "
            "manuscript. Items used as predictors are additionally hand-verified "
            "and recorded in config/predictor_allowlist.yaml."
        ),
        "items": entries,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    book = build(args.pdf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        yaml.safe_dump(book, fh, sort_keys=False, allow_unicode=True, width=100)

    n = len(book["items"])
    flagged = sum(1 for v in book["items"].values() if v["needs_manual_review"])
    print(f"Wrote {n} items to {args.out} ({flagged} flagged for manual review)")


if __name__ == "__main__":
    main()
