"""Design-matrix construction and the leakage guard.

Answers editor comment 2 ("Rule out outcome leakage. Provide the complete list
of candidate predictors and explain how variables directly or indirectly
related to the mathematics score or proficiency category were excluded").

THE DEFECT THIS REPLACES
------------------------
``codes/data_preparation.py:253`` built the design matrix as::

    categorical_columns = df.select_dtypes(include=['object']).columns
    X = df_exp.drop(columns=list(categorical_columns) + ['label']).fillna(0)

``select_dtypes(include=['object'])`` removes only *string* columns. The
outcome ``math_score`` is float64 and survived, as did all ten
``PV1MATH``-``PV10MATH`` columns, every reading and science plausible value,
``W_FSTUWT``, the 80 BRR replicate weights, and ``STRATUM``. Because the label
is a hard threshold on ``math_score``, a decision stump on that one column
scores 100%. This is the whole of the reported AUC = 1.0000.

DESIGN PRINCIPLES
-----------------
1. **Allowlist, not denylist.** A column enters ``X`` only if it is named in
   ``config/predictor_allowlist.yaml``. A new PISA column added upstream
   cannot silently become a predictor.
2. **Belt and braces.** The denylist (``FORBIDDEN_PATTERNS``) runs *as well*,
   so a mistake in the allowlist is still caught.
3. **Hard error.** Violations raise :class:`LeakageError`. Nothing prints a
   warning and continues.
4. **Every entry point.** ``assert_no_leakage`` is called in data preparation,
   feature selection, model fitting and explanation -- not once at the top.
5. **Empirical screen.** Any single variable achieving AUC > 0.95 alone is
   flagged and cannot enter without recorded sign-off.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class LeakageError(RuntimeError):
    """Raised when a forbidden column reaches the design matrix.

    This is deliberately an exception and not a warning. Editor comment 1 and
    2 both turn on the guarantee that no outcome-derived or design variable
    can influence model development.
    """


# ---------------------------------------------------------------------------
# Denylist. Every pattern is anchored with fullmatch semantics via
# ``_compiled()``. Additions must be justified in CHANGELOG_REVISION.md.
# ---------------------------------------------------------------------------
FORBIDDEN_PATTERNS: List[str] = [
    # --- outcome and everything derived from it -------------------------
    r"PV\d+MATH",           # the ten mathematics plausible values: the outcome
    r"PV\d+READ",           # reading PVs: same conditioning model, contaminated
    r"PV\d+SCIE",           # science PVs: same
    r"PV\d+[A-Z]+\d*",      # every subscale PV (MATH SHAPE/SPACE, READ TEXT, ...)
    r"math_score",          # the mean of the math PVs -- the direct leak
    r"math_category",       # the three-level label
    r"label",               # the binary label
    r"proficiency.*",
    r"MATH", r"READ", r"SCIE",
    # --- survey design and weights --------------------------------------
    r"W_FST.*",             # W_FSTUWT and W_FSTURWT1..80
    r"W_SCH.*",
    r"SENWT",
    r"STRATUM.*",           # sampling stratum, incl. one-hot STRATUM_ESP90xx
    r"VER_.*",
    # --- identifiers and administrative metadata ------------------------
    r"CNTSCHID", r"CNTSTUID", r"CNTRYID",
    r"CNT", r"CYC", r"NatCen", r"SUBNATIO", r"OECD",
    r"ADMINMODE", r"LANGTEST", r"BOOKID", r"Region",
    r"Unnamed.*", r"^\s*$",
    # --- the negative control (Stage 2.5): diagnostic use only ----------
    r"NOISE_CONTROL",
    # --- OECD-derived composites. Not the outcome, but constructed by the
    #     OECD from overlapping questionnaire material and, in the case of
    #     ESCS, used in the PISA conditioning model that generates the
    #     plausible values. Forbidden until individually justified.
    r"ESCS", r"HISEI", r"HISCED", r"PARED", r"MISCED", r"FISCED",
    r"BSMJ", r"WEALTH", r"HOMEPOS", r"CULTPOSS", r"HEDRES", r"ICTRES",
]

#: Reason strings surfaced in the exception, keyed by the pattern that fired.
_PATTERN_REASONS: Dict[str, str] = {
    r"PV\d+MATH": "mathematics plausible value -- the outcome is a threshold on these",
    r"PV\d+READ": "reading plausible value -- shares the PISA conditioning model with the outcome",
    r"PV\d+SCIE": "science plausible value -- shares the PISA conditioning model with the outcome",
    r"PV\d+[A-Z]+\d*": "plausible value (subscale) -- outcome-derived",
    r"math_score": "the outcome variable itself (mean of PV1MATH..PV10MATH)",
    r"math_category": "the three-level proficiency label",
    r"label": "the binary classification target",
    r"W_FST.*": "survey weight (final or BRR replicate) -- design variable",
    r"SENWT": "senate weight -- design variable",
    r"STRATUM.*": "explicit sampling stratum -- design variable, not a predictor",
    r"CNTSCHID": "school identifier -- the CV grouping variable (PSU)",
    r"CNTSTUID": "student identifier",
    r"NOISE_CONTROL": "Gaussian negative control -- diagnostic runs only",
    r"ESCS": "OECD composite used in the PISA conditioning model for the plausible values",
}

_EMPIRICAL_AUC_CEILING = 0.95

#: Compiled once. The audit noted the old code recompiled 30 regexes per column.
_COMPILED: List[tuple] = []


_COMPILED = [(p, re.compile(p, re.IGNORECASE)) for p in FORBIDDEN_PATTERNS]


def _normalise(col: str) -> str:
    """Canonicalise a column name before matching.

    The audit of this file found the denylist to be case-sensitive with no
    whitespace tolerance, so every one of these slipped through:

        pv1math, PV1math, MATH_SCORE, ' math_score', 'math_score\n',
        math_score_z, log_math_score, PV1MATH_imputed, PV1MATH_1.0,
        math_category_Low, missingindicator_math_score

    Normalising to upper case with surrounding whitespace stripped, and
    matching derived forms explicitly, closes those holes. The allowlist is
    still the primary defence; this is the belt to its braces, and a belt with
    holes is not a belt.
    """
    return re.sub(r"\s+", "", str(col)).upper()


#: Wrappers that indicate a DERIVED form of a forbidden variable. A transformed
#: outcome is still the outcome.
_DERIVED_AFFIXES = (
    "MISSINGINDICATOR_", "LOG_", "SQRT_", "Z_", "STD_", "SCALED_", "NORM_",
    "IMPUTED_", "BINNED_", "RANK_", "PCT_",
)
_DERIVED_SUFFIXES = (
    "_Z", "_STD", "_SCALED", "_NORM", "_MEAN", "_RANK", "_PCT", "_BIN",
    "_IMPUTED", "_LOG", "_CAT", "_QUANTILE",
)


def _candidate_forms(col: str) -> List[str]:
    """The name plus every plausible derived/encoded form to test."""
    c = _normalise(col)
    forms = {c}
    for pre in _DERIVED_AFFIXES:
        if c.startswith(pre):
            forms.add(c[len(pre):])
    for suf in _DERIVED_SUFFIXES:
        if c.endswith(suf):
            forms.add(c[: -len(suf)])
    # One-hot / binned encodings: MATH_CATEGORY_LOW, PV1MATH_1.0, ST004D01T_2.0
    head = re.split(r"_(?=[^_]*$)", c)[0] if "_" in c else c
    forms.add(head)
    return [f for f in forms if f]


def _compiled() -> List[tuple]:
    return [(p, re.compile(p, re.IGNORECASE)) for p in FORBIDDEN_PATTERNS]


def _reason_for(col: str) -> str:
    for form in _candidate_forms(col):
        for pat, rx in _COMPILED:
            if rx.fullmatch(form):
                base = _PATTERN_REASONS.get(pat, f"matches forbidden pattern {pat!r}")
                if form != _normalise(col):
                    return f"{base} (derived form of {form})"
                return base
    return "matches a forbidden pattern"


def forbidden_columns(columns: Iterable[str]) -> List[str]:
    """Return every column matching a forbidden pattern.

    Matching is case-insensitive, whitespace-insensitive, and covers derived
    and one-hot-encoded forms. A standardised, log-transformed, imputed or
    one-hot-encoded outcome is still the outcome.
    """
    return [
        c for c in columns
        if any(rx.fullmatch(f) for f in _candidate_forms(c) for _, rx in _COMPILED)
    ]


def assert_no_leakage(
    X: pd.DataFrame | np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    *,
    where: str = "unspecified",
) -> None:
    """Raise :class:`LeakageError` if any forbidden column is present.

    Call this at *every* entry point. ``where`` is echoed in the message so a
    failure identifies the stage that let the column through.
    """
    if isinstance(X, pd.DataFrame):
        cols = list(X.columns)
    elif feature_names is not None:
        cols = list(feature_names)
    else:
        raise ValueError(
            "assert_no_leakage needs a DataFrame or an explicit feature_names "
            "list. Positional feature indices are exactly how the original "
            "pipeline lost track of which columns it was using "
            "(AUDIT_REPORT.md section C.2)."
        )

    bad = forbidden_columns(cols)
    if bad:
        detail = "\n".join(f"    - {c}: {_reason_for(c)}" for c in sorted(bad)[:40])
        more = "" if len(bad) <= 40 else f"\n    ... and {len(bad) - 40} more"
        raise LeakageError(
            f"{len(bad)} forbidden column(s) reached the design matrix at "
            f"stage {where!r}:\n{detail}{more}\n"
            "These are outcome, outcome-derived, or survey-design variables. "
            "This is the defect the editor identified "
            "(codes/data_preparation.py:253)."
        )


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------
@dataclass
class AllowlistEntry:
    name: str
    block: str
    label: str
    scale: str
    justification: str
    source: str = "student_questionnaire"
    encoding: str = "ordinal"
    single_auc: Optional[float] = None
    signed_off_by: Optional[str] = None


class Allowlist:
    """The versioned, justified list of candidate predictors.

    Serialised in ``config/predictor_allowlist.yaml`` with one row per
    variable: PISA item code, official codebook label, response scale and a
    one-line justification for inclusion. That file is the supplementary table
    answering the editor's demand for "the complete list of candidate
    predictors".
    """

    def __init__(self, entries: Dict[str, AllowlistEntry], meta: Optional[Dict] = None):
        self.entries = entries
        self.meta = meta or {}

    @classmethod
    def load(cls, path: Path) -> "Allowlist":
        path = Path(path)
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
        entries = {}
        for name, rec in (doc.get("predictors") or {}).items():
            entries[name] = AllowlistEntry(
                name=name,
                block=rec.get("block", ""),
                label=rec.get("label", ""),
                scale=rec.get("scale", ""),
                justification=rec.get("justification", ""),
                source=rec.get("source", "student_questionnaire"),
                encoding=rec.get("encoding", "ordinal"),
                single_auc=rec.get("single_auc"),
                signed_off_by=rec.get("signed_off_by"),
            )
        alw = cls(entries, doc.get("meta", {}))
        alw.validate()
        return alw

    def validate(self) -> None:
        """An allowlist entry that is also forbidden is a contradiction."""
        contradictions = forbidden_columns(self.entries)
        if contradictions:
            raise LeakageError(
                "predictor_allowlist.yaml lists columns that the denylist "
                f"forbids: {sorted(contradictions)}. Resolve the contradiction "
                "before running anything."
            )
        missing = [n for n, e in self.entries.items() if not e.justification.strip()]
        if missing:
            raise ValueError(
                f"Allowlist entries without a justification: {sorted(missing)}. "
                "Editor comment 2 requires a stated reason for every candidate "
                "predictor."
            )

    @property
    def names(self) -> List[str]:
        return list(self.entries)

    def expand(self, columns: Iterable[str]) -> List[str]:
        """Match allowlisted names against actual columns, including one-hots.

        ``ST004D01T`` in the allowlist matches ``ST004D01T_1.0`` and
        ``ST004D01T_2.0`` in a one-hot-encoded frame. Matching is by NAME,
        never by position -- the positional indexing at
        ``feature_selection.ipynb`` cell 20 is what made the original
        "selected features" unverifiable.
        """
        cols = [str(c) for c in columns]
        out: List[str] = []
        for name in self.names:
            exact = [c for c in cols if c == name]
            onehot = [c for c in cols if c.startswith(name + "_")]
            out.extend(exact or onehot)
        return list(dict.fromkeys(out))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(e) for e in self.entries.values()])


# ---------------------------------------------------------------------------
# Design-matrix construction
# ---------------------------------------------------------------------------
def build_design_matrix(
    df: pd.DataFrame,
    allowlist: Allowlist,
    *,
    strict: bool = True,
    where: str = "build_design_matrix",
    include_noise_control: bool = False,
) -> pd.DataFrame:
    """Build ``X`` from an allowlist and refuse to return a leaking matrix.

    Parameters
    ----------
    df:
        The analytic frame. May contain outcome, design and identifier
        columns; they are simply not selected.
    allowlist:
        Loaded :class:`Allowlist`.
    strict:
        When True (default and the only supported setting for published runs),
        a missing allowlisted column is an error rather than a silent drop.
    include_noise_control:
        Diagnostic runs only (Stage 2.5). Appends ``noise_control`` *after* the
        guard, as an explicitly labelled negative control, and logs loudly.
    """
    wanted = allowlist.expand(df.columns)

    if strict:
        unmatched = [
            n for n in allowlist.names
            if not any(c == n or c.startswith(n + "_") for c in df.columns)
        ]
        if unmatched:
            raise KeyError(
                f"Allowlisted predictors absent from the data: {sorted(unmatched)}. "
                "Either the wrong file was loaded or the allowlist is stale. "
                "Refusing to proceed with a silently smaller feature set."
            )

    X = df.loc[:, wanted].copy()
    assert_no_leakage(X, where=where)

    if include_noise_control:
        if "noise_control" not in df.columns:
            raise KeyError("noise_control requested but absent from the frame.")
        logger.warning(
            "Appending noise_control to X as a NEGATIVE CONTROL (diagnostic "
            "run). Results from this matrix must never be reported as "
            "substantive findings."
        )
        X["noise_control"] = df["noise_control"].to_numpy()

    return X


# ---------------------------------------------------------------------------
# Empirical leakage screen
# ---------------------------------------------------------------------------
def single_variable_auc_screen(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    ceiling: float = _EMPIRICAL_AUC_CEILING,
    allowlist: Optional[Allowlist] = None,
    raise_on_flag: bool = True,
) -> pd.DataFrame:
    """AUC of each candidate predictor on its own, as a leakage tripwire.

    A single questionnaire item cannot separate proficiency groups almost
    perfectly. Anything above ``ceiling`` is either the outcome in disguise or
    a design artefact. Flagged variables cannot enter without explicit
    sign-off recorded as ``signed_off_by`` in the allowlist YAML.

    Returns a frame with one row per variable, sorted by ``auc`` descending.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    rows = []
    for col in X.columns:
        v = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(v)
        if ok.sum() < 10 or len(np.unique(y[ok])) < 2 or len(np.unique(v[ok])) < 2:
            rows.append({"variable": col, "auc": np.nan, "n_used": int(ok.sum())})
            continue
        a = roc_auc_score(y[ok], v[ok])
        rows.append(
            {
                "variable": col,
                # Direction-free separability: a perfectly *inverted* predictor
                # leaks just as much as a perfectly aligned one.
                "auc": max(a, 1.0 - a),
                "auc_signed": a,
                "n_used": int(ok.sum()),
            }
        )

    out = pd.DataFrame(rows).sort_values("auc", ascending=False, na_position="last")
    out["flagged"] = out["auc"] > ceiling
    out["ceiling"] = ceiling

    flagged = out.loc[out["flagged"], "variable"].tolist()
    if flagged:
        signed = set()
        if allowlist is not None:
            signed = {
                n for n, e in allowlist.entries.items() if e.signed_off_by
            }
        unsigned = [f for f in flagged if f not in signed]
        msg = (
            f"Single-variable AUC screen flagged {len(flagged)} variable(s) "
            f"above {ceiling}: {flagged}. A lone questionnaire item does not "
            "separate proficiency groups this well; treat as leakage until "
            "proven otherwise."
        )
        if unsigned and raise_on_flag:
            raise LeakageError(
                msg + f" Unsigned-off: {unsigned}. Record a signed_off_by entry "
                "in config/predictor_allowlist.yaml to override."
            )
        logger.warning(msg)

    return out.reset_index(drop=True)


def classify_columns(
    columns: Iterable[str],
    codebook=None,
) -> pd.DataFrame:
    """Classify every dataset column, with a justification (Stage 1, task 2).

    Categories: ``outcome``, ``outcome-derived``, ``design/weight``,
    ``identifier``, ``negative-control``, ``candidate predictor``.
    Written to ``results/audit/column_classification.csv``.
    """
    rows = []
    for c in columns:
        c = str(c)
        if re.fullmatch(r"math_score|math_category|label", c):
            cat, why = "outcome", "the target, or the variable the target thresholds"
        elif re.fullmatch(r"PV\d+MATH", c):
            cat, why = "outcome", "mathematics plausible value; the outcome is derived from these"
        elif re.fullmatch(r"PV\d+[A-Z]+\d*", c):
            cat, why = (
                "outcome-derived",
                "plausible value from the same PISA conditioning model as the outcome",
            )
        elif re.fullmatch(r"W_FST.*|W_SCH.*|SENWT", c):
            cat, why = "design/weight", "survey weight (final or BRR replicate)"
        elif re.fullmatch(r"STRATUM.*", c):
            cat, why = "design/weight", "explicit sampling stratum"
        elif re.fullmatch(r"CNTSCHID|CNTSTUID|CNTRYID", c):
            cat, why = "identifier", "unit identifier; CNTSCHID is the PSU used for grouping"
        elif re.fullmatch(r"CNT|CYC|NatCen|SUBNATIO|OECD|VER_.*|ADMINMODE|LANGTEST|BOOKID|Region", c):
            cat, why = "identifier", "administrative metadata"
        elif c == "noise_control":
            cat, why = "negative-control", "injected Gaussian noise; diagnostic benchmark only"
        else:
            cat = "candidate predictor"
            why = "questionnaire item; admissible if allowlisted with a codebook citation"
        row = {"column": c, "classification": cat, "justification": why}
        if codebook is not None:
            e = codebook.describe(c)
            row["codebook_label"] = e.label
            row["codebook_source"] = e.source
        rows.append(row)
    return pd.DataFrame(rows)
