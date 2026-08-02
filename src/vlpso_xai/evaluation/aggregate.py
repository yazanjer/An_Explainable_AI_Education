"""Aggregation of per-fold predictions into reportable estimates.

Closes audit finding **M2**:

    "Notebook 05 pools predictions across methods, repeats and PVs into one
    AUC and one CI, duplicating each student up to 35 times, and globs
    checkpoints without a config-fingerprint filter."

WHAT WAS WRONG
--------------
The notebook did this::

    preds = pd.concat([pd.read_parquet(p) for p in
                       glob.glob(str(cfg.paths.checkpoints / "*_preds.parquet"))],
                      ignore_index=True)
    for task, sub in preds.groupby("task"):
        ci = cluster_bootstrap_ci(sub.y_true, sub.y_score, sub.group, ...)

Four separate errors compound in those five lines.

1. **The glob is unfiltered by configuration fingerprint.** The checkpoint
   directory holds files from every run that has ever touched it. At the time
   of writing it contains two distinct fingerprints (``cfg-2a228fe0c5`` and
   ``cfg-350625c08f``) plus legacy files written before fingerprinting
   existed. Those are different partitions of different samples. Concatenating
   them produces a number that describes no single analysis.

2. **Pooling across methods.** ``none``, ``chi2``, ``mrmr``, ``bpso`` and
   ``vlpso`` are different estimators. Their union is not an estimator.

3. **Pooling across repeats.** Each repeat is a *re-partition of the same
   students*. Five repeats put every student in the frame five times. The
   students are not five independent samples, so the bootstrap sees 5N rows
   where the design supports N, and the interval collapses.

4. **Pooling across plausible values.** Ten PVs put every student in the frame
   ten more times, and — worse — with *different labels*, because 69.7% of
   students change proficiency category across PVs. PV variance is the largest
   single component of uncertainty here (FMI = 0.563). Pooling does not
   propagate it; it launders it into a spuriously tight interval.

   Combined, 5 repeats x 10 PVs x 5 methods duplicates each student up to 250
   times, hence the audit's "up to 35" being, if anything, generous.

THE CORRECT HIERARCHY
---------------------
There are three levels, and only the first is a pooling operation::

    level 1  folds within (task, method, pv, repeat)
             -> CONCATENATE. The K outer folds partition the sample, so each
                student appears exactly once. This is the out-of-fold
                prediction vector. AUC and a school-clustered BCa bootstrap
                variance are computed here, and nowhere else.

    level 2  repeats within (task, method, pv)
             -> AVERAGE the per-repeat AUCs. Repeats are re-partitions of one
                sample, not new samples. Their spread is partition noise; it
                is reported as a range, never added to the sampling variance.

    level 3  plausible values within (task, method)
             -> RUBIN'S RULES. U = mean within-PV sampling variance from the
                clustered bootstrap; B = between-PV variance. This is the only
                step that yields a publishable CI, and it is the step that
                carries the PV measurement error the pooled version discarded.

    methods  -> NEVER combined. Compared, with
                :func:`vlpso_xai.evaluation.effect_size.contrast_table`.

Everything in this module refuses rather than guesses: an ambiguous checkpoint
set raises, a partial fold set raises, and an overlapping fold set raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.outcome import rubin_combine
from .metrics import classification_metrics, cluster_bootstrap_ci

__all__ = [
    "CheckpointSet",
    "parse_checkpoint_name",
    "load_fold_predictions",
    "oof_predictions",
    "auc_by_repeat",
    "aggregate_task_method",
    "aggregation_table",
]

#: ``fold_task-low_vs_high_pv-1_method-none_rep-0_fold-3_cfg-350625c08f_preds.parquet``
_NAME_RE = re.compile(
    r"^fold_"
    r"task-(?P<task>.+?)_"
    r"pv-(?P<pv>\d+)_"
    r"method-(?P<method>.+?)_"
    r"rep-(?P<rep>\d+)_"
    r"fold-(?P<fold>\d+)"
    r"(?:_cfg-(?P<cfg>[0-9a-f]+))?"
    r"_preds\.parquet$"
)

KEY_COLS = ("task", "pv", "method", "rep", "fold")


def parse_checkpoint_name(path: Path | str) -> Optional[Dict[str, object]]:
    """Extract ``(task, pv, method, rep, fold, cfg_fingerprint)`` from a filename.

    Returns ``None`` for anything that is not a prediction checkpoint, so the
    caller can distinguish "not mine" from "malformed".

    The fingerprint lives in the filename rather than the payload for files
    written before :func:`vlpso_xai.evaluation.nested_cv.run_nested_cv` began
    embedding ``cfg_fingerprint`` as a column. Files with no fingerprint at all
    are pre-fingerprinting legacy artefacts and are reported with
    ``cfg_fingerprint = None``; :class:`CheckpointSet` treats them as a
    distinct, unusable configuration rather than silently folding them in.
    """
    m = _NAME_RE.match(Path(path).name)
    if m is None:
        return None
    g = m.groupdict()
    return {
        "task": g["task"],
        "pv": int(g["pv"]),
        "method": g["method"],
        "rep": int(g["rep"]),
        "fold": int(g["fold"]),
        "cfg_fingerprint": g["cfg"],
        "path": Path(path),
    }


@dataclass
class CheckpointSet:
    """The prediction checkpoints for exactly one configuration fingerprint."""

    index: pd.DataFrame
    fingerprint: Optional[str]

    @property
    def tasks(self) -> List[str]:
        return sorted(self.index["task"].unique())

    @property
    def methods(self) -> List[str]:
        return sorted(self.index["method"].unique())

    def describe(self) -> pd.DataFrame:
        """Fold inventory per (task, method, pv, rep). Print this before use."""
        g = (
            self.index.groupby(["task", "method", "pv", "rep"])["fold"]
            .agg(n_folds="count", folds=lambda s: sorted(s.tolist()))
            .reset_index()
        )
        return g


def _scan(checkpoint_dir: Path | str) -> pd.DataFrame:
    rows = [
        r for r in (parse_checkpoint_name(p)
                    for p in sorted(Path(checkpoint_dir).glob("*_preds.parquet")))
        if r is not None
    ]
    if not rows:
        raise FileNotFoundError(
            f"No '*_preds.parquet' checkpoints in {checkpoint_dir}. Nested CV "
            "must be run with a checkpoint directory before aggregation; "
            "summary metrics alone cannot produce a clustered bootstrap."
        )
    return pd.DataFrame(rows)


def load_fold_predictions(
    checkpoint_dir: Path | str,
    *,
    cfg_fingerprint: Optional[str] = None,
    task: Optional[str | Sequence[str]] = None,
    method: Optional[str | Sequence[str]] = None,
    require_unique_fingerprint: bool = True,
) -> CheckpointSet:
    """Index the prediction checkpoints for ONE configuration.

    Parameters
    ----------
    cfg_fingerprint:
        The 10-character configuration hash to keep. If ``None`` and the
        directory holds more than one, this **raises** and lists the candidates
        with their fold counts. It does not pick the newest, the largest or the
        first — every one of those heuristics has a failure mode in which a
        stale run is reported as current, which is the exact class of error
        this module exists to prevent.
    require_unique_fingerprint:
        Escape hatch for tests only. Setting it ``False`` permits a mixed set
        and is never correct for a reported number.

    Raises
    ------
    ValueError
        If the directory holds several fingerprints and none was specified.
    """
    idx = _scan(checkpoint_dir)

    fps = idx["cfg_fingerprint"].where(idx["cfg_fingerprint"].notna(), None)
    distinct = sorted({f for f in fps if f is not None})
    has_legacy = bool(fps.isna().any())

    if cfg_fingerprint is None and require_unique_fingerprint:
        n_variants = len(distinct) + (1 if has_legacy else 0)
        if n_variants > 1:
            counts = (
                idx.assign(cfg=idx["cfg_fingerprint"].fillna("<legacy, unfingerprinted>"))
                .groupby("cfg")
                .size()
                .sort_values(ascending=False)
            )
            listing = "\n".join(f"    {k}: {v} folds" for k, v in counts.items())
            raise ValueError(
                "The checkpoint directory holds more than one configuration "
                "fingerprint. These are different partitions of possibly "
                "different samples and must not be combined.\n"
                f"{listing}\n"
                "Pass cfg_fingerprint=... to choose one. Legacy files without "
                "a fingerprint predate configuration hashing and cannot be "
                "matched to a config; delete them or move them aside."
            )
        cfg_fingerprint = distinct[0] if distinct else None

    if cfg_fingerprint is not None:
        idx = idx[idx["cfg_fingerprint"] == cfg_fingerprint]
        if idx.empty:
            raise ValueError(
                f"No checkpoints with fingerprint {cfg_fingerprint!r}. "
                f"Available: {distinct or '<none>'}."
            )

    for col, val in (("task", task), ("method", method)):
        if val is not None:
            want = [val] if isinstance(val, str) else list(val)
            idx = idx[idx[col].isin(want)]
            if idx.empty:
                raise ValueError(f"No checkpoints with {col} in {want}.")

    return CheckpointSet(index=idx.reset_index(drop=True), fingerprint=cfg_fingerprint)


def oof_predictions(
    cset: CheckpointSet,
    task: str,
    method: str,
    pv: int,
    rep: int,
    *,
    expected_folds: Optional[int] = None,
) -> pd.DataFrame:
    """Out-of-fold predictions for one (task, method, pv, repeat).

    This is **level 1** — the only legitimate concatenation. The outer folds of
    a single repeat partition the analytic sample, so each student contributes
    exactly one prediction.

    Two structural checks run here because both failure modes are silent:

    * a *missing* fold shrinks the sample and biases the AUC toward whichever
      schools survived;
    * an *overlapping* fold means the partition assumption is false, which
      would duplicate students exactly as the pooled version did.

    Overlap is detected on ``(group, position-within-group)`` composites rather
    than a student identifier, because the prediction checkpoints store the
    school id but no student id. This catches whole-fold duplication — the
    realistic failure, e.g. a checkpoint written twice — but cannot detect a
    partial overlap of individual students. The stronger check belongs in
    ``run_nested_cv``, where ``assert_group_disjoint`` already enforces it at
    split time.
    """
    sub = cset.index[
        (cset.index["task"] == task)
        & (cset.index["method"] == method)
        & (cset.index["pv"] == pv)
        & (cset.index["rep"] == rep)
    ].sort_values("fold")
    if sub.empty:
        raise ValueError(f"No folds for task={task!r} method={method!r} pv={pv} rep={rep}.")

    folds = sub["fold"].tolist()
    if len(set(folds)) != len(folds):
        raise ValueError(f"Duplicate fold indices {folds} for {task}/{method}/pv{pv}/rep{rep}.")
    if expected_folds is not None and len(folds) != expected_folds:
        raise ValueError(
            f"{task}/{method}/pv{pv}/rep{rep} has {len(folds)} folds, expected "
            f"{expected_folds} (found {folds}). A partial repeat is not an "
            "out-of-fold prediction set and must not be scored as one."
        )
    if folds != list(range(len(folds))):
        raise ValueError(
            f"Fold indices {folds} are not contiguous from 0 for "
            f"{task}/{method}/pv{pv}/rep{rep}."
        )

    frames = []
    for _, r in sub.iterrows():
        d = pd.read_parquet(r["path"])
        # Runs from the current code embed the fingerprint as a column. When it
        # is present it must agree with the filename: a mismatch means a file
        # was renamed or copied between checkpoint directories, and the
        # filename-based index is then lying about what the frame contains.
        if "cfg_fingerprint" in d.columns and r["cfg_fingerprint"] is not None:
            inside = set(d["cfg_fingerprint"].dropna().unique())
            if inside and inside != {r["cfg_fingerprint"]}:
                raise ValueError(
                    f"{Path(r['path']).name} is named for configuration "
                    f"{r['cfg_fingerprint']!r} but contains {sorted(inside)!r}. "
                    "The file has been renamed or copied; its provenance is "
                    "unknown and it must not be scored."
                )
        d["fold"] = r["fold"]
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)

    # Whole-fold duplication check (see docstring for what this does not cover).
    seen: Dict[object, int] = {}
    dup_folds = set()
    for f, sch in zip(out["fold"], out["group"]):
        key = sch
        if key in seen and seen[key] != f:
            dup_folds.add((seen[key], f))
        seen.setdefault(key, f)
    if dup_folds:
        raise ValueError(
            f"Schools appear in more than one outer fold of "
            f"{task}/{method}/pv{pv}/rep{rep}: fold pairs {sorted(dup_folds)}. "
            "The outer split was not school-grouped, or checkpoints from "
            "different partitions have been mixed. Either way the pooled "
            "prediction vector is invalid."
        )
    return out


def auc_by_repeat(
    cset: CheckpointSet,
    task: str,
    method: str,
    pv: int,
    *,
    metric: str = "auc",
    n_resamples: int = 2000,
    bootstrap_method: str = "BCa",
    weighted: bool = False,
    alpha: float = 0.05,
    expected_folds: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """**Level 2** — one row per repeat, each from a disjoint OOF vector.

    The school-clustered bootstrap variance is computed *per repeat* and later
    averaged into Rubin's ``U``. Computing it on a repeat-pooled frame would
    resample schools that appear several times, which is the tighter-interval
    failure again in a different costume.
    """
    reps = sorted(
        cset.index[
            (cset.index["task"] == task)
            & (cset.index["method"] == method)
            & (cset.index["pv"] == pv)
        ]["rep"].unique()
    )
    rows = []
    for rep in reps:
        d = oof_predictions(cset, task, method, pv, rep, expected_folds=expected_folds)
        w = d["sample_weight"].to_numpy() if weighted else None
        if w is not None and not np.isfinite(w).all():
            raise ValueError(
                f"weighted=True but sample_weight contains non-finite values in "
                f"{task}/{method}/pv{pv}/rep{rep}. The nested-CV run stored NaN, "
                "meaning it was executed without survey weights."
            )
        ci = cluster_bootstrap_ci(
            d["y_true"].to_numpy(), d["y_score"].to_numpy(), d["group"].to_numpy(),
            metric=metric, n_resamples=n_resamples, method=bootstrap_method,
            alpha=alpha, sample_weight=w, random_state=random_state + int(rep),
        )
        rows.append({
            "task": task, "method": method, "pv": int(pv), "rep": int(rep),
            "n_students": int(len(d)), "n_schools": int(d["group"].nunique()),
            "n_folds": int(d["fold"].nunique()),
            metric: float(ci["estimate"]),
            "boot_sd": float(ci.get("boot_sd", np.nan)),
            "sampling_variance": float(ci.get("boot_sd", np.nan)) ** 2,
            "ci_low": float(ci["ci_low"]), "ci_high": float(ci["ci_high"]),
            "bootstrap": ci["method"], "weighted": bool(weighted),
        })
    return pd.DataFrame(rows)


def aggregate_task_method(
    cset: CheckpointSet,
    task: str,
    method: str,
    *,
    metric: str = "auc",
    n_resamples: int = 2000,
    bootstrap_method: str = "BCa",
    weighted: bool = False,
    alpha: float = 0.05,
    expected_folds: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, object]:
    """**Level 3** — Rubin's rules across plausible values.

    ``U`` is the mean of the per-PV design-based sampling variances (school
    clustered bootstrap, averaged over repeats within PV). ``B`` is the
    variance of the per-PV point estimates. The returned ``fmi`` is the share
    of total variance contributed by the plausible values, i.e. by measurement
    error in proficiency rather than by sampling.

    With a single PV, ``B`` is undefined; :func:`rubin_combine` warns and
    returns ``B = 0``. The result is then a quick-mode diagnostic and is
    flagged ``single_pv = True``. It must not be published: the handover's
    headline CI [0.8621, 0.8849] carries FMI 0.563, so a single-PV interval
    understates the SE by roughly a factor of 1.5.
    """
    per_rep = pd.concat(
        [
            auc_by_repeat(
                cset, task, method, pv, metric=metric, n_resamples=n_resamples,
                bootstrap_method=bootstrap_method, weighted=weighted, alpha=alpha,
                expected_folds=expected_folds, random_state=random_state,
            )
            for pv in sorted(
                cset.index[
                    (cset.index["task"] == task) & (cset.index["method"] == method)
                ]["pv"].unique()
            )
        ],
        ignore_index=True,
    )
    if per_rep.empty:
        raise ValueError(f"No results for task={task!r} method={method!r}.")

    per_pv = (
        per_rep.groupby("pv")
        .agg(
            estimate=(metric, "mean"),
            repeat_sd=(metric, lambda s: float(s.std(ddof=1)) if s.size > 1 else 0.0),
            repeat_min=(metric, "min"),
            repeat_max=(metric, "max"),
            sampling_variance=("sampling_variance", "mean"),
            n_repeats=(metric, "size"),
            n_students=("n_students", "max"),
            n_schools=("n_schools", "max"),
        )
        .reset_index()
    )

    rr = rubin_combine(
        per_pv["estimate"].to_numpy(),
        per_pv["sampling_variance"].to_numpy(),
        alpha=alpha,
    )
    out = {
        "task": task, "method": method, "metric": metric,
        "n_pv": int(len(per_pv)),
        "n_repeats": int(per_rep["rep"].nunique()),
        "n_folds_per_repeat": int(per_rep["n_folds"].max()),
        "n_students": int(per_pv["n_students"].max()),
        "n_schools": int(per_pv["n_schools"].max()),
        "single_pv": bool(len(per_pv) < 2),
        "weighted": bool(weighted),
        "cfg_fingerprint": cset.fingerprint,
        "repeat_spread": float(per_pv["repeat_sd"].mean()),
    }
    out.update(rr.as_dict())
    return {"summary": out, "per_pv": per_pv, "per_repeat": per_rep}


def aggregation_table(
    cset: CheckpointSet,
    *,
    metric: str = "auc",
    tasks: Optional[Iterable[str]] = None,
    methods: Optional[Iterable[str]] = None,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Run the full hierarchy for every (task, method) present.

    Returns three frames — ``summary`` (level 3, one row per task x method),
    ``per_pv`` (level 2 averaged) and ``per_repeat`` (level 1). All three are
    written to ``results/`` so that every published number has a file behind
    it, per the project constraint that no number may be reported which a
    script did not produce.
    """
    tasks = list(tasks) if tasks is not None else cset.tasks
    methods = list(methods) if methods is not None else cset.methods

    summaries, pvs, reps = [], [], []
    for t in tasks:
        for m in methods:
            present = cset.index[(cset.index["task"] == t) & (cset.index["method"] == m)]
            if present.empty:
                continue
            res = aggregate_task_method(cset, t, m, metric=metric, **kwargs)
            summaries.append(res["summary"])
            pvs.append(res["per_pv"].assign(task=t, method=m))
            reps.append(res["per_repeat"])
    if not summaries:
        raise ValueError("No (task, method) combination produced a result.")
    return {
        "summary": pd.DataFrame(summaries),
        "per_pv": pd.concat(pvs, ignore_index=True),
        "per_repeat": pd.concat(reps, ignore_index=True),
    }
