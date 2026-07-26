"""Official PISA 2018 codebook lookup.

Answers editor comment 7: "Provide official PISA item descriptions and coding
for every interpreted variable. Do not assign educational meanings to coded
features without documentary support."

Three layers, in increasing order of authority:

1. ``config/codebook_pisa2018_student.yaml`` -- auto-extracted from the OECD
   instrument PDF by ``scripts/build_codebook.py``. Reliable question stems,
   sometimes-wrapped item labels.
2. ``config/codebook_overrides.yaml`` -- hand-transcribed entries for every
   item the analysis interprets, plus the documented status of every
   non-questionnaire column.
3. SPSS value-label metadata read from ``CY07_MSU_STU_QQQ.sav`` -- the
   authoritative *as-released* coding, which differs from the printed
   instrument for recoded blocks such as ST019.

``describe()`` is the only sanctioned way to attach a human-readable meaning to
a column name. ``require_documented()`` raises if a variable is about to be
interpreted without a source, so an undocumented gloss cannot reach the
manuscript.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class UndocumentedVariableError(RuntimeError):
    """Raised when a variable would be interpreted without a codebook source."""


@dataclass
class CodebookEntry:
    item: str
    label: str = ""
    block: str = ""
    question_stem: str = ""
    response_categories: Dict[Any, str] = field(default_factory=dict)
    scale_type: str = ""
    direction: str = ""
    page_pdf: Optional[int] = None
    source: str = ""
    reverse_scored: Optional[bool] = None
    construct_note: str = ""
    coding_note: str = ""
    manuscript_error: str = ""

    def citation(self) -> str:
        """A citable one-liner for the manuscript and supplementary tables."""
        where = f", p. {self.page_pdf}" if self.page_pdf else ""
        return (
            f"{self.item}: \"{self.label}\" "
            f"(stem: \"{self.question_stem.strip()}\"); "
            f"OECD PISA 2018 Student Questionnaire{where}"
        )

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["response_categories"] = json.dumps(self.response_categories, ensure_ascii=False)
        return d


class Codebook:
    """Merged, versioned PISA item lookup."""

    def __init__(
        self,
        extracted: Optional[Dict] = None,
        overrides: Optional[Dict] = None,
        spss_value_labels: Optional[Dict[str, Dict]] = None,
        spss_column_labels: Optional[Dict[str, str]] = None,
    ):
        self._extracted = (extracted or {}).get("items", {})
        self._meta = (extracted or {}).get("source", {})
        self._ov = overrides or {}
        self._ov_items = self._ov.get("items", {})
        self._ov_blocks = self._ov.get("blocks", {})
        self._nonq = self._ov.get("non_questionnaire", {})
        self._spss_values = spss_value_labels or {}
        self._spss_labels = spss_column_labels or {}
        self._cache: Dict[str, CodebookEntry] = {}

    # -- construction ----------------------------------------------------
    @classmethod
    def load(
        cls,
        config_dir: Path,
        spss_meta: Any = None,
        cycle: int = 2018,
    ) -> "Codebook":
        config_dir = Path(config_dir)
        extracted_path = config_dir / f"codebook_pisa{cycle}_student.yaml"
        overrides_path = config_dir / "codebook_overrides.yaml"

        extracted = {}
        if extracted_path.exists():
            with open(extracted_path) as fh:
                extracted = yaml.safe_load(fh) or {}
        else:
            logger.warning(
                "Auto-extracted codebook not found at %s. Run "
                "scripts/build_codebook.py. Falling back to overrides only.",
                extracted_path,
            )

        overrides = {}
        if overrides_path.exists():
            with open(overrides_path) as fh:
                overrides = yaml.safe_load(fh) or {}

        value_labels, col_labels = {}, {}
        if spss_meta is not None:
            value_labels = _spss_value_labels(spss_meta)
            col_labels = dict(
                zip(
                    getattr(spss_meta, "column_names", []),
                    getattr(spss_meta, "column_labels", []),
                )
            )

        return cls(extracted, overrides, value_labels, col_labels)

    # -- lookup ----------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        base = self.base_name(name)
        return (
            base in self._ov_items
            or base in self._extracted
            or base in self._spss_labels
            or self.non_questionnaire_role(base) is not None
        )

    @staticmethod
    def base_name(name: str) -> str:
        """Strip a one-hot suffix: ``ST004D01T_1.0`` -> ``ST004D01T``."""
        return re.split(r"_(?=[^A-Za-z]|$)", name, maxsplit=1)[0]

    def describe(self, name: str) -> CodebookEntry:
        """Return the merged codebook entry for a (possibly encoded) column."""
        if name in self._cache:
            return self._cache[name]

        base = self.base_name(name)
        entry = CodebookEntry(item=name)
        sources: List[str] = []

        auto = self._extracted.get(base)
        if auto:
            entry.block = auto.get("block", "")
            entry.question_stem = auto.get("question_stem", "")
            entry.label = auto.get("item_label", "")
            entry.page_pdf = auto.get("page_pdf")
            sources.append("instrument_pdf")

        ov = self._ov_items.get(base)
        if ov:
            entry.block = ov.get("block", entry.block)
            entry.label = ov.get("label", entry.label)
            entry.reverse_scored = ov.get("reverse_scored", entry.reverse_scored)
            entry.manuscript_error = ov.get("manuscript_error", "") or ""
            sources.append("hand_verified")

        blk = self._ov_blocks.get(entry.block)
        if blk:
            entry.question_stem = blk.get("question_stem", entry.question_stem)
            entry.scale_type = blk.get("scale_type", "")
            entry.direction = blk.get("direction", "")
            entry.page_pdf = blk.get("page_pdf", entry.page_pdf)
            entry.construct_note = blk.get("construct_note", "") or ""
            entry.coding_note = blk.get("coding_note", "") or ""
            cats = blk.get("response_categories") or blk.get(
                "response_categories_instrument"
            )
            if cats:
                entry.response_categories = dict(cats)

        if base in self._spss_labels and self._spss_labels[base]:
            if not entry.label:
                entry.label = self._spss_labels[base]
            sources.append("spss_metadata")
        if base in self._spss_values:
            # Authoritative as-released coding wins over the printed instrument.
            entry.response_categories = dict(self._spss_values[base])

        entry.source = "+".join(dict.fromkeys(sources)) or "undocumented"
        self._cache[name] = entry
        return entry

    def label(self, name: str) -> str:
        e = self.describe(name)
        return e.label or name

    def non_questionnaire_role(self, name: str) -> Optional[str]:
        """Role of a non-questionnaire column (outcome/design/identifier/...)."""
        for key, rec in self._nonq.items():
            for pattern in _expand_key(key):
                if re.fullmatch(pattern, name):
                    return rec.get("role")
        return None

    def require_documented(self, names: List[str]) -> None:
        """Raise unless every name resolves to a documented source.

        Called before any table or figure that attaches meaning to a variable.
        """
        missing = [
            n for n in names
            if self.describe(n).source == "undocumented"
            and self.non_questionnaire_role(self.base_name(n)) is None
        ]
        if missing:
            raise UndocumentedVariableError(
                "Refusing to interpret variables with no codebook source: "
                f"{sorted(missing)}. Add them to config/codebook_overrides.yaml "
                "with a page citation, or drop them from the interpretation. "
                "(Editor comment 7.)"
            )

    def to_frame(self, names: List[str]):
        """Supplementary-table export: one row per interpreted variable."""
        import pandas as pd

        return pd.DataFrame([self.describe(n).to_row() for n in names])


def _expand_key(key: str) -> List[str]:
    """Turn a codebook key such as ``PV1MATH..PV10MATH`` into regex patterns."""
    key = key.strip()
    if " / " in key:
        return [re.escape(k.strip()) for k in key.split(" / ")]
    m = re.match(r"^([A-Za-z_]+)(\d+)([A-Za-z_]*)\.\.\1(\d+)\3$", key)
    if m:
        prefix, _, suffix, _ = m.groups()
        return [rf"{re.escape(prefix)}\d+{re.escape(suffix)}"]
    return [re.escape(key)]


def _spss_value_labels(meta: Any) -> Dict[str, Dict]:
    """Pull ``{column: {code: label}}`` out of pyreadstat metadata."""
    out: Dict[str, Dict] = {}
    v2l = getattr(meta, "variable_to_label", {}) or {}
    labelsets = getattr(meta, "value_labels", {}) or {}
    for col, setname in v2l.items():
        if setname in labelsets:
            out[col] = dict(labelsets[setname])
    return out
