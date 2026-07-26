"""Emit every manuscript table as both .tex and .csv. No number typed by hand.

The LaTeX writer is hand-rolled rather than ``DataFrame.to_latex``, which
routes through pandas Styler and therefore requires jinja2 >= 3.1.2. Pinning a
templating engine to emit a tabular environment is not a trade worth making in
a reproducibility-critical pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

_TEX_ESCAPES = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
}


def escape_tex(s: object) -> str:
    text = "" if s is None else str(s)
    return re.sub(
        "|".join(re.escape(k) for k in sorted(_TEX_ESCAPES, key=len, reverse=True)),
        lambda m: _TEX_ESCAPES[m.group()],
        text,
    )


def _fmt(v: object, float_format: str) -> str:
    if isinstance(v, float):
        return "" if pd.isna(v) else float_format % v
    return escape_tex(v)


def to_latex(
    df: pd.DataFrame,
    *,
    caption: str = "",
    label: str = "",
    float_format: str = "%.4f",
) -> str:
    """A plain booktabs tabular. Deterministic, dependency-free."""
    align = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) else "l"
                    for c in df.columns)
    header = " & ".join(escape_tex(c.replace("_", " ")) for c in df.columns)
    body = " \\\\\n".join(
        " & ".join(_fmt(v, float_format) for v in row)
        for row in df.itertuples(index=False, name=None)
    )
    return (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{escape_tex(caption)}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{align}}}\n\\toprule\n"
        f"{header} \\\\\n\\midrule\n{body} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def emit(
    df: pd.DataFrame,
    name: str,
    outdir: Path,
    *,
    caption: str = "",
    label: Optional[str] = None,
    float_format: str = "%.4f",
    columns: Optional[Sequence[str]] = None,
) -> dict:
    """Write ``name.csv`` and ``name.tex`` side by side."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        df = df.loc[:, list(columns)]
    csv_path = outdir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    tex_path = outdir / f"{name}.tex"
    tex_path.write_text(
        to_latex(df, caption=caption or name.replace("_", " ").title(),
                 label=label or f"tab:{name}", float_format=float_format)
    )
    return {"csv": csv_path, "tex": tex_path, "rows": len(df)}
