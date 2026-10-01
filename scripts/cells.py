#!/usr/bin/env python3
"""Round-2 analyses: manifest, run, reconcile, aggregate.

    python scripts/cells.py manifest                       # write results/cells_manifest.csv
    python scripts/cells.py run --workers 32               # compute every missing cell
    python scripts/cells.py run --kinds perm --limit 3     # a few cells of one kind
    python scripts/cells.py run --cell 'sel|fold=0|method=vlpso|pv=1|rep=0|task=low_vs_high'
    python scripts/cells.py reconcile                      # expected vs completed, per kind
    python scripts/cells.py aggregate                      # every round-2 table

Workers are processes forked after the analytic frames are built, so the data
are read once. Each cell is independent and idempotent (see
src/vlpso_xai/experiments/cells.py); stopping and restarting the run loses at
most the cells in flight. ``--synthetic`` runs on generated data with the real
column layout, for tests and timing; its results are never written to the
real results directory.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logger = logging.getLogger("cells")
_STATE = {}


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _work(row):
    from vlpso_xai.experiments.cells import run_cell
    t0 = time.perf_counter()
    st = run_cell(row, _STATE["frames"], _STATE["r2"], _STATE["results"], extra=_STATE["extra"])
    return row["cell_id"], st, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["manifest", "run", "reconcile", "aggregate", "publish"])
    ap.add_argument("--dest", default=None)
    ap.add_argument("--kinds", default="eda,brr,ext,perm,shap,vlstab,sel")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cell", default=None)
    ap.add_argument("--shard", default=None, help="i/n: run every n-th pending cell starting at i")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--r3-dir", default=os.environ.get("R3_DIR"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--config", default="default")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    from vlpso_xai.config import load_config, set_global_seeds
    from vlpso_xai.experiments.cells import is_done, load_r2, manifest, reconcile

    r2 = load_r2(ROOT / "config" / "revision_r2.yaml")
    cfg = load_config(args.config, root=Path(os.environ.get("VLPSO_PROJECT_ROOT", ROOT)))
    set_global_seeds(cfg.seed)
    results = Path(args.results) if args.results else (
        ROOT / "results_synthetic" if args.synthetic else cfg.paths.results)
    kinds = args.kinds.split(",")
    man = manifest(r2, kinds)
    results.mkdir(parents=True, exist_ok=True)

    if args.command == "manifest":
        man.to_csv(results / "cells_manifest.csv", index=False)
        print(man.groupby("kind").size().to_string())
        return 0
    if args.command == "reconcile":
        rec = reconcile(results, manifest(r2))
        rec.to_csv(results / "cells_reconciliation.csv", index=False)
        print(rec.to_string(index=False))
        return 0
    if args.command == "publish":
        from vlpso_xai.experiments.publish import publish
        n = publish(results, Path(args.dest))
        print(f"published {n} files to {args.dest}")
        return 0
    if args.command == "aggregate":
        from vlpso_xai.experiments.aggregate import aggregate_all
        aggregate_all(results, r2, r3_dir=args.r3_dir)
        return 0

    # ---- run ---------------------------------------------------------------
    if args.cell:
        man = man[man.cell_id == args.cell]
        if man.empty:
            raise SystemExit(f"cell {args.cell!r} is not in the manifest")
    pending = [r for r in man.to_dict("records") if not is_done(results, r)]
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        pending = pending[i::n]
    if args.limit:
        pending = pending[: args.limit]
    logger.info("%d cells in manifest for %s; %d pending", len(man), kinds, len(pending))
    if not pending:
        return 0

    if args.synthetic:
        from vlpso_xai.experiments.data import synthetic_frames
        frames = synthetic_frames(n_schools=int(os.environ.get("SYN_SCHOOLS", 120)),
                                  per=int(os.environ.get("SYN_PER", 25)))
    else:
        from vlpso_xai.experiments.data import build_frames
        frames = build_frames(cfg)
    _STATE.update(frames=frames, r2=r2, results=results,
                  extra={"r3_dir": args.r3_dir, "cells_dir": str(results / "cells" / "shap")})

    deadline = time.time() + 3600 * args.max_hours if args.max_hours else None
    n_done = n_fail = 0
    t0 = time.time()
    if args.workers <= 1:
        it = map(_work, pending)
    else:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(args.workers, initializer=_init_worker, maxtasksperchild=50)
        it = pool.imap_unordered(_work, pending, chunksize=1)
    for cid, st, sec in it:
        n_done += st == "done"
        n_fail += st == "failed"
        logger.info("%-7s %6.1fs  %s  [%d done, %d failed, %.2f h]", st, sec, cid, n_done, n_fail,
                    (time.time() - t0) / 3600)
        if deadline and time.time() > deadline:
            logger.warning("max-hours reached; stopping (completed cells are kept)")
            if args.workers > 1:
                pool.terminate()
            break
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
