"""Environment-aware configuration.

Replaces ``codes/config.py``, which hard-coded
``/content/drive/MyDrive/educational-disparities-analysis`` and raised an
exception on any other path (AUDIT_REPORT.md section I). That made the
original repository impossible to run locally, in CI, or on any Colab
account other than the authors'.

Resolution order for the project root:

1. ``VLPSO_PROJECT_ROOT`` environment variable, if set.
2. Google Drive mount, if running under Colab and the folder exists.
3. Repository root (directory containing requirements.txt / pyproject.toml / .git).
4. ``./`` with a loud warning.

No magic numbers live in this module. Every grid, seed, budget and threshold
comes from ``config/default.yaml`` (or ``config/quick.yaml``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

#: Drive folders searched, in order, when running under Colab. The first that
#: exists wins. The repository name comes first so a fresh clone works, with
#: the shorter legacy name kept for existing setups.
DRIVE_CANDIDATES = (
    Path("/content/drive/MyDrive/An_Explainable_AI_Education"),
    Path("/content/drive/MyDrive/vlpso-xai-pisa"),
)
DRIVE_DEFAULT = DRIVE_CANDIDATES[0]
_ROOT_MARKERS = ("requirements.txt", "pyproject.toml", ".git")


def in_colab() -> bool:
    """True when executing inside a Google Colab runtime."""
    return "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG") is not None


def _repo_root() -> Optional[Path]:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if any((parent / marker).exists() for marker in _ROOT_MARKERS):
            return parent
    return None


def resolve_project_root() -> Path:
    """Resolve the project root without hard-coding any absolute path."""
    env = os.environ.get("VLPSO_PROJECT_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        logger.info("Project root from VLPSO_PROJECT_ROOT: %s", root)
        return root

    if in_colab():
        for candidate in DRIVE_CANDIDATES:
            if candidate.exists():
                logger.info("Project root from Google Drive mount: %s", candidate)
                return candidate

    repo = _repo_root()
    if repo is not None:
        logger.info("Project root from repository layout: %s", repo)
        return repo

    cwd = Path(".").resolve()
    logger.warning(
        "Could not resolve a project root from the environment, a Drive mount, or "
        "repository markers. Falling back to the working directory: %s. "
        "Set VLPSO_PROJECT_ROOT to silence this warning.",
        cwd,
    )
    return cwd


@dataclass(frozen=True)
class Paths:
    """All filesystem locations, derived from a single root."""

    root: Path
    data_raw: Path
    data_processed: Path
    results: Path
    models: Path
    checkpoints: Path
    figures: Path
    tables: Path

    @classmethod
    def from_root(cls, root: Path) -> "Paths":
        root = Path(root)
        return cls(
            root=root,
            data_raw=root / "data" / "raw",
            data_processed=root / "data" / "processed",
            results=root / "results",
            models=root / "models",
            checkpoints=root / "results" / "checkpoints",
            figures=root / "results" / "figures",
            tables=root / "results" / "tables",
        )

    def mkdirs(self) -> "Paths":
        """Create every generated directory. Raw data is never created here."""
        for p in (
            self.data_processed,
            self.results,
            self.models,
            self.checkpoints,
            self.figures,
            self.tables,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class Config:
    """Parsed YAML config plus resolved paths."""

    raw: Dict[str, Any]
    paths: Paths
    config_path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def section(self, *keys: str) -> Any:
        """Traverse nested keys, raising a clear error on a missing one."""
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                raise KeyError(
                    "Missing config key %r (failed at %r) in %s"
                    % (".".join(keys), k, self.config_path)
                )
            node = node[k]
        return node

    @property
    def seed(self) -> int:
        return int(self.section("reproducibility", "master_seed"))

    @property
    def seeds(self) -> List[int]:
        """The list of distinct seeds used for repeated runs (editor comment 4)."""
        n = int(self.section("reproducibility", "n_seeds"))
        return [self.seed + i for i in range(n)]

    def hash(self) -> str:
        """Stable SHA256 of the config contents, recorded in the manifest."""
        blob = json.dumps(self.raw, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()


def load_config(name: str = "default", root: Optional[Path] = None) -> Config:
    """Load ``config/<name>.yaml`` relative to the project root."""
    root = Path(root) if root is not None else resolve_project_root()

    candidate = Path(name)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        config_path = candidate.resolve()
    else:
        config_path = root / "config" / ("%s.yaml" % name)
        if not config_path.exists():
            alt = Path(__file__).resolve().parents[2] / "config" / ("%s.yaml" % name)
            if alt.exists():
                config_path = alt
    if not config_path.exists():
        raise FileNotFoundError(
            "Config %r not found at %s. Set VLPSO_PROJECT_ROOT to the repository "
            "root, or pass an explicit path." % (name, config_path)
        )

    with open(config_path) as fh:
        raw = yaml.safe_load(fh)

    override = raw.get("paths", {}).get("root")
    paths_root = Path(override).expanduser() if override else root
    return Config(raw=raw, paths=Paths.from_root(paths_root), config_path=config_path)


def set_global_seeds(seed: int) -> None:
    """Seed every RNG we can reach. Called at the top of every entry point."""
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except Exception:  # pragma: no cover - optional dependency
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - optional dependency
        pass


def environment_report() -> Dict[str, Any]:
    """Record the environment so a run can be reproduced or explained."""
    import platform

    report: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "in_colab": in_colab(),
        "project_root": str(resolve_project_root()),
        "packages": {},
    }
    for mod in (
        "numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm",
        "shap", "lime", "tensorflow", "pyreadstat", "boruta", "skrebate",
    ):
        try:
            m = __import__(mod)
            report["packages"][mod] = getattr(m, "__version__", "unknown")
        except Exception:
            report["packages"][mod] = None
    return report
