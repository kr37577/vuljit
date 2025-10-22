"""Plotting utilities for RQ3 analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:  # pragma: no cover - matplotlib optional
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

__all__ = ["plot_additional_builds_boxplot"]


def plot_additional_builds_boxplot(
    project_metrics: pd.DataFrame,
    output_dir: str,
) -> Optional[str]:
    """Generate a boxplot of scheduled additional builds per strategy."""

    if plt is None or project_metrics.empty:
        return None

    output_path = Path(output_dir) / "additional_builds_boxplot.png"
    pivot = project_metrics.pivot_table(
        index="project",
        columns="strategy",
        values="scheduled_builds",
        aggfunc="sum",
    )
    if pivot.empty:
        return None

    plt.figure(figsize=(10, 6))
    pivot.plot(
        kind="box",
        ax=plt.gca(),
        title="Scheduled Additional Builds per Strategy (Project-Level)",
        grid=True,
    )
    plt.ylabel("Scheduled additional builds")
    plt.xlabel("Strategy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return str(output_path)

