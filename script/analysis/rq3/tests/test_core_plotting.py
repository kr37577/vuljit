from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from RQ3.core import plotting
from RQ3.core.plotting import plot_additional_builds_boxplot


@pytest.mark.skipif(plotting.plt is None, reason="matplotlib not available")
def test_plot_additional_builds_boxplot_creates_file(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "project": ["alpha", "beta"],
            "strategy": ["s1", "s1"],
            "scheduled_builds": [1.0, 2.0],
        }
    )
    path = plot_additional_builds_boxplot(df, str(tmp_path))
    assert path is not None
    assert Path(path).is_file()


def test_plot_additional_builds_boxplot_returns_none_without_backend(monkeypatch, tmp_path) -> None:
    df = pd.DataFrame({"project": [], "strategy": [], "scheduled_builds": []})
    monkeypatch.setattr(plotting, "plt", None, raising=False)
    assert plot_additional_builds_boxplot(df, str(tmp_path)) is None
