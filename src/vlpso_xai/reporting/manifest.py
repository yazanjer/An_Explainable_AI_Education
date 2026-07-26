"""Artefact manifest: SHA256, provenance and config hash for every result.

Answers the reproducibility requirement behind editor comment 1. The submitted
repository had no record of which code or configuration produced which number,
and `validation.py:188,205` "validated" results against hard-coded expected
answers -- a circular check that cannot fail informatively.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(repo: Optional[Path] = None) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo or Path.cwd()),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def build_manifest(
    results_dir: Path,
    *,
    config_hash: Optional[str] = None,
    generating_script: Optional[str] = None,
    repo: Optional[Path] = None,
    environment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Hash every artefact under ``results_dir``."""
    results_dir = Path(results_dir)
    entries: List[Dict[str, Any]] = []
    for p in sorted(results_dir.rglob("*")):
        if not p.is_file() or p.name == "manifest.json":
            continue
        entries.append({
            "path": str(p.relative_to(results_dir)),
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
        })
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repo),
        "config_hash": config_hash,
        "generating_script": generating_script,
        "environment": environment,
        "n_artefacts": len(entries),
        "artefacts": entries,
    }


def write_manifest(results_dir: Path, **kwargs) -> Path:
    m = build_manifest(results_dir, **kwargs)
    out = Path(results_dir) / "manifest.json"
    out.write_text(json.dumps(m, indent=2))
    return out


def compare_manifests(a: Path, b: Path) -> Dict[str, Any]:
    """Determinism check (Stage 10 check 5): two runs, identical hashes."""
    ma, mb = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    da = {e["path"]: e["sha256"] for e in ma["artefacts"]}
    db = {e["path"]: e["sha256"] for e in mb["artefacts"]}
    common = set(da) & set(db)
    differing = sorted(p for p in common if da[p] != db[p])
    return {
        "identical": not differing and set(da) == set(db),
        "n_common": len(common),
        "n_differing": len(differing),
        "differing": differing,
        "only_in_a": sorted(set(da) - set(db)),
        "only_in_b": sorted(set(db) - set(da)),
    }
