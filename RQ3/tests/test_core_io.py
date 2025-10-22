from __future__ import annotations

from pathlib import Path

import pandas as pd

from RQ3.core import ensure_directory, ensure_parent_directory, parse_build_counts_csv


def test_ensure_directory_creates_path(tmp_path) -> None:
    target = tmp_path / "nested" / "dir"
    resolved = ensure_directory(target)
    assert Path(resolved).is_dir()


def test_ensure_parent_directory_creates_parent(tmp_path) -> None:
    file_path = tmp_path / "outputs" / "result.csv"
    resolved = ensure_parent_directory(file_path)
    assert Path(resolved).parent.is_dir()


def test_parse_build_counts_csv_reads_mapping(tmp_path) -> None:
    csv_path = tmp_path / "counts.csv"
    df = pd.DataFrame({"project": ["alpha", "beta"], "builds_per_day": [2.2, 3.9]})
    df.to_csv(csv_path, index=False)

    mapping = parse_build_counts_csv(csv_path)
    assert mapping == {"alpha": 2, "beta": 4}
