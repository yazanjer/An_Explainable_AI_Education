"""Cell manifest, identity, seeding and atomic, idempotent execution.

Handover section 5: every result is a cell; nothing let one cell be requested
on its own. Here:

* :func:`manifest` enumerates every cell of every analysis up front from
  ``config/revision_r2.yaml``. The manifest is the work queue, the progress
  ledger and the provenance record.
* :func:`cell_id` is a canonical string; :func:`cell_seed` derives the cell's
  random seed from it (never from wall-clock or worker index), so a cell gives
  the same result whichever worker runs it and in whatever order.
* :func:`run_cell` computes one cell and writes ``<results>/cells/<kind>/<id>/``
  atomically: everything goes to a temporary sibling directory that is renamed
  into place only after ``meta.json`` is written. A worker killed mid-cell
  leaves no partial result. Re-running a completed cell is a no-op.
* The output directory name carries a fingerprint of the cell's configuration
  section and schema version, so a changed setting can never resume a stale
  result (the checkpoint-reuse trap of round 1).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

KIND_ORDER = ["eda", "brr", "ext", "perm", "shap", "vlstab", "sel"]


def load_r2(path: Optional[Path] = None) -> Dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[3] / "config" / "revision_r2.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)


def cell_id(kind: str, **params) -> str:
    return kind + "|" + "|".join(f"{k}={params[k]}" for k in sorted(params))


def parse_cell_id(cid: str) -> Dict[str, Any]:
    kind, *rest = cid.split("|")
    out: Dict[str, Any] = {"kind": kind}
    for kv in rest:
        k, v = kv.split("=", 1)
        out[k] = int(v) if v.lstrip("-").isdigit() else v
    return out


def cell_seed(cid: str) -> int:
    return int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def section_fingerprint(r2: Dict[str, Any], kind: str) -> str:
    blob = json.dumps({"schema": r2["meta"]["schema_version"], "seed": r2["seed"],
                       "section": r2.get(kind, {})}, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


def safe_name(cid: str) -> str:
    return cid.replace("|", "__").replace("=", "-")


def manifest(r2: Dict[str, Any], kinds: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Every cell, ordered so that any prefix is a balanced design.

    Selector cells are ordered PV-major: all methods, tasks, repeats and folds
    for PV1, then PV2, and so on. A run stopped early therefore holds complete,
    matched designs for the first PVs rather than a ragged subset.
    """
    kinds = list(kinds or KIND_ORDER)
    rows: List[Dict[str, Any]] = []

    def add(kind, **p):
        cid = cell_id(kind, **p)
        rows.append({"kind": kind, "cell_id": cid, "seed": cell_seed(cid),
                     "fingerprint": section_fingerprint(r2, kind), **p})

    if "eda" in kinds:
        add("eda", scope="spain_analytic")
    if "brr" in kinds:
        b = r2["brr"]
        for t in b["tasks"]:
            for pv in b["pvs"]:
                add("brr", task=t, pv=pv)
    if "ext" in kinds:
        e = r2["ext"]
        for t in e["tasks"]:
            for pv in e["pvs"]:
                add("ext", task=t, pv=pv)
    if "perm" in kinds:
        p = r2["perm"]
        add("perm", null="observed", draw=0)
        for i in range(p["n_unrestricted"]):
            add("perm", null="unrestricted", draw=i)
        for i in range(p["n_within_school"]):
            add("perm", null="within_school", draw=i)
    if "shap" in kinds:
        s = r2["shap"]
        for pv in s["pvs"]:
            for t in s["tasks"]:
                for f in s["folds"]:
                    add("shap", task=t, pv=pv, fold=f)
    if "vlstab" in kinds:
        v = r2["vlstab"]
        for f in v["folds"]:
            for m in v["seed_methods"]:
                for sd in v["seeds"]:
                    add("vlstab", variant=f"seed_{m}", fold=f, swarm_seed=sd)
            for name in v["variants"]:
                add("vlstab", variant=name, fold=f, swarm_seed=0)
    if "sel" in kinds:
        s = r2["sel"]
        for pv in s["pvs"]:
            for rep in s["repeats"]:
                for t in s["tasks"]:
                    for f in s["folds"]:
                        for m in s["methods"]:
                            add("sel", task=t, pv=pv, rep=rep, fold=f, method=m)
    return pd.DataFrame(rows)


def out_dir(results: Path, row: Dict[str, Any]) -> Path:
    return Path(results) / "cells" / row["kind"] / f"{safe_name(row['cell_id'])}__{row['fingerprint']}"


def is_done(results: Path, row: Dict[str, Any]) -> bool:
    return (out_dir(results, row) / "meta.json").exists()


def write_atomic(target: Path, tables: Dict[str, pd.DataFrame], meta: Dict[str, Any]) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".tmp_{target.name}_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    for name, df in tables.items():
        df.to_parquet(tmp / f"{name}.parquet", index=False)
    with open(tmp / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True, default=str)
    if target.exists():                       # another worker finished first
        shutil.rmtree(tmp)
        return
    os.rename(tmp, target)


def run_cell(row: Dict[str, Any], frames, r2: Dict[str, Any], results: Path,
             *, extra: Optional[Dict[str, Any]] = None) -> str:
    """Compute one cell if it is not already complete. Returns a status string."""
    from . import kinds as K

    target = out_dir(results, row)
    if (target / "meta.json").exists():
        return "skipped"
    fn: Callable = getattr(K, f"run_{row['kind']}")
    t0 = time.perf_counter()
    try:
        kw = (extra or {}) if row["kind"] == "brr" else {}
        tables, meta = fn(row, frames, r2, **kw)
    except Exception as exc:                  # recorded, never silently swallowed
        err = Path(results) / "cells" / "_errors"
        err.mkdir(parents=True, exist_ok=True)
        (err / f"{safe_name(row['cell_id'])}.txt").write_text(traceback.format_exc())
        logger.error("cell %s FAILED: %s", row["cell_id"], exc)
        return "failed"
    meta.update({"cell_id": row["cell_id"], "kind": row["kind"], "seed": row["seed"],
                 "fingerprint": row["fingerprint"],
                 "seconds": round(time.perf_counter() - t0, 2)})
    write_atomic(target, tables, meta)
    return "done"


def reconcile(results: Path, man: pd.DataFrame) -> pd.DataFrame:
    """Expected vs. completed, per kind. Written to the results as provenance."""
    done = man.apply(lambda r: is_done(results, r.to_dict()), axis=1)
    out = man.assign(done=done).groupby("kind")["done"].agg(["size", "sum"])
    out.columns = ["expected", "completed"]
    out["missing"] = out["expected"] - out["completed"]
    return out.reset_index()
