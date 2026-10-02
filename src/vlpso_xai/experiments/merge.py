"""Rebuild a local cells directory from several published pod branches.

Each pod publishes its completed cells as concatenated per-kind tables with a
``cell_id`` column plus ``meta.jsonl``. This splits them back into the
one-directory-per-cell layout so :func:`aggregate.aggregate_all` runs on the
union of all pods exactly as it would on one machine. Duplicate cells (a cell
computed by two pods) are checked for identical results and kept once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .cells import safe_name, write_atomic
from .publish import PUBLISHABLE


def merge_published(sources: Iterable[Path], results: Path) -> pd.DataFrame:
    rows = []
    for src in sources:
        src = Path(src)
        for kind in PUBLISHABLE:
            mf = src / "cells" / kind / "meta.jsonl"
            if not mf.exists():
                continue
            metas = pd.DataFrame([json.loads(l) for l in mf.read_text().splitlines() if l.strip()])
            tables = {t: pd.read_parquet(src / "cells" / kind / f"{t}.parquet")
                      for t in PUBLISHABLE[kind] if (src / "cells" / kind / f"{t}.parquet").exists()}
            for m in metas.to_dict("records"):
                cid = m["cell_id"]
                target = results / "cells" / kind / f"{safe_name(cid)}__{m['fingerprint']}"
                tabs = {t: df[df.cell_id == cid].drop(columns="cell_id").reset_index(drop=True)
                        for t, df in tables.items() if (df.cell_id == cid).any()}
                if (target / "meta.json").exists():
                    old = json.loads((target / "meta.json").read_text())
                    same = all(abs(float(old.get(k, 0)) - float(m.get(k, 0))) < 1e-12
                               for k in ("auc", "auc_fold_weighted", "auc_weighted") if k in m)
                    # sel/vlstab keep their AUC and subset in result.parquet, not meta
                    rp = target / "result.parquet"
                    if same and "result" in tabs and rp.exists():
                        a, b = pd.read_parquet(rp), tabs["result"]
                        same = (abs(float(a.auc.iloc[0]) - float(b.auc.iloc[0])) < 1e-12
                                and a.get("selected", pd.Series([None])).iloc[0]
                                == b.get("selected", pd.Series([None])).iloc[0])
                    rows.append({"source": str(src), "kind": kind, "cell_id": cid, "status": "duplicate" if same else "CONFLICT"})
                    continue
                write_atomic(target, tabs, {k: v for k, v in m.items()})
                rows.append({"source": str(src), "kind": kind, "cell_id": cid, "status": "merged"})
    return pd.DataFrame(rows)
