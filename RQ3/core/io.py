"""Common file-system helpers and default path management for RQ3."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Union

import pandas as pd

PathLike = Union[str, os.PathLike]

_CORE_DIR = Path(__file__).resolve().parent
_RQ3_ROOT = _CORE_DIR.parent
_REPO_ROOT = _RQ3_ROOT.parent

# Centralised defaults used by the phase4/phase5 CLI scripts and utilities.
DEFAULTS: MutableMapping[str, Any] = {
    # Additional-build simulation defaults
    "phase5.output_dir": _RQ3_ROOT / "simulation_outputs",
    "phase5.predictions_root": _REPO_ROOT / "outputs" / "results" / "random_forest",
    "phase5.detection_table": _REPO_ROOT / "rq3_dataset" / "detection_time_results.csv",
    "phase5.build_counts": _REPO_ROOT / "rq3_dataset" / "project_build_counts.csv",
    "phase5.detection_window_days": 0,
    # Minimal additional-build simulation defaults
    "phase5_minimal.output": _RQ3_ROOT / "minimal_simulation_summary.csv",
    "phase5_minimal.predictions_root": _REPO_ROOT / "outputs" / "results" / "random_forest",
    # Phase 4 analysis defaults
    "phase4.output_dir": _RQ3_ROOT / "phase4_outputs",
    "phase4.predictions_root": _REPO_ROOT / "outputs" / "results" / "random_forest",
    # Build timeline defaults
    "timeline.data_dir": _REPO_ROOT / "data",
    "timeline.build_counts": _REPO_ROOT / "rq3_dataset" / "project_build_counts.csv",
    "timeline.output_dir": _RQ3_ROOT / "timeline_outputs" / "build_timelines",
    "timeline.default_builds_per_day": 1,
}


def _coerce_path(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    return Path(os.fspath(value))


def normalise_path(value: PathLike) -> str:
    """Return an absolute, normalised string path."""

    return str(_coerce_path(value).resolve())


def ensure_directory(path: PathLike) -> str:
    """Ensure ``path`` exists as a directory and return the normalised path."""

    directory = _coerce_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return normalise_path(directory)


def ensure_parent_directory(path: PathLike) -> str:
    """Ensure the parent directory of ``path`` exists (useful for file outputs)."""

    file_path = _coerce_path(path)
    if file_path.parent:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    return normalise_path(file_path)


def _convert_env_value(raw: str, reference: Optional[Any]) -> Any:
    if reference is None or isinstance(reference, (Path, str)):
        return normalise_path(raw)
    if isinstance(reference, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(reference, int):
        return int(raw)
    if isinstance(reference, float):
        return float(raw)
    return raw


def resolve_default(key: str, fallback: Optional[Any] = None) -> Any:
    """Return a default configuration value or ``fallback`` if the key is absent."""

    env_key = f"RQ3_DEFAULT_{key.replace('.', '_').upper()}"
    if env_key in os.environ:
        reference = DEFAULTS.get(key, fallback)
        return _convert_env_value(os.environ[env_key], reference)

    if key in DEFAULTS:
        value = DEFAULTS[key]
        if isinstance(value, Path):
            return normalise_path(value)
        return value
    if fallback is not None:
        return fallback
    raise KeyError(f"Unknown default key: {key}")


def resolve_default_path(key: str, *extra: PathLike, create: bool = False) -> str:
    """Resolve a default path and optionally append sub-paths."""

    base = resolve_default(key)
    path = _coerce_path(base)
    for component in extra:
        path = path / _coerce_path(component)
    if create:
        ensure_directory(path)
    return normalise_path(path)


def load_detection_table(path: PathLike) -> pd.DataFrame:
    """Load the detection baseline CSV and return a cleaned DataFrame."""

    df = pd.read_csv(path)
    project_col = "project" if "project" in df.columns else "package_name"
    df["project"] = df[project_col].astype(str).str.strip()
    df = df[df["project"] != ""]
    if "detection_time_days" in df.columns:
        df["detection_time_days"] = pd.to_numeric(
            df["detection_time_days"], errors="coerce"
        )
    return df.dropna(subset=["project", "detection_time_days"])


def load_build_counts(path: PathLike) -> pd.DataFrame:
    """Load per-project build cadence information."""

    df = pd.read_csv(path)
    df["project"] = df.get("project", "").astype(str).str.strip()
    df = df[df["project"] != ""]
    df["builds_per_day"] = pd.to_numeric(df.get("builds_per_day"), errors="coerce")
    return df.dropna(subset=["builds_per_day"])


def parse_build_counts_csv(path: PathLike) -> Dict[str, int]:
    """Return a simple project->build-count mapping for CLI usage."""

    df = load_build_counts(path)
    mapping = df.set_index("project")["builds_per_day"].round().astype(int).to_dict()
    return {project: max(count, 0) for project, count in mapping.items()}
