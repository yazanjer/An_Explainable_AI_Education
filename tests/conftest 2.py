"""Shared fixtures. Every test runs on SYNTHETIC data.

No PISA microdata is required, so the suite runs in CI where the OECD file
cannot be redistributed.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config"


@pytest.fixture(scope="session")
def allowlist():
    from vlpso_xai.data.features import Allowlist
    return Allowlist.load(CONFIG / "predictor_allowlist.yaml")


@pytest.fixture
def synthetic(allowlist):
    """Clustered synthetic data mimicking the PISA design.

    200 schools x 25 students. A school-level effect makes students within a
    school correlated, so a test that ignores clustering will visibly differ
    from one that respects it.
    """
    rng = np.random.default_rng(20260725)
    n_schools, per_school = 200, 25
    n = n_schools * per_school
    school = np.repeat(np.arange(n_schools), per_school)
    school_effect = rng.normal(0, 1.0, n_schools)[school]

    names = allowlist.names
    X = pd.DataFrame(
        {c: rng.integers(1, 5, n).astype(float) for c in names}
    )
    signal = 0.7 * X[names[0]] + 0.5 * X[names[1]] - 0.4 * X[names[2]]
    logit = 0.6 * school_effect + 0.35 * (signal - signal.mean()) / signal.std()
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    return {
        "X": X,
        "y": y,
        "groups": school,
        "weights": rng.gamma(4, 50, n),
        "n_schools": n_schools,
    }
