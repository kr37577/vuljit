"""Backward-compatible wrapper for the build timeline CLI."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Tuple


def _import_cli_modules(package_name: str) -> Tuple[ModuleType, ModuleType]:
    """Import CLI and timeline modules from the specified package path."""

    cli_module = importlib.import_module(f"{package_name}.cli.build_timeline_cli")
    timeline_module = importlib.import_module(f"{package_name}.core.timeline")
    return cli_module, timeline_module


if __package__ in (None, ""):
    _wrapper_dir = Path(__file__).resolve().parent
    _analysis_root = _wrapper_dir.parent
    _repo_root = _analysis_root.parent
    _project_root = _repo_root.parent

    for path in ( _analysis_root, _repo_root, _project_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    _cli_module: ModuleType
    _timeline_module: ModuleType
    _cli_module, _timeline_module = _import_cli_modules("vuljit.analysis.research_question3")

    main = _cli_module.main  # type: ignore[attr-defined]
    parse_args = _cli_module.parse_args  # type: ignore[attr-defined]
    build_timeline = _timeline_module.build_timeline  # type: ignore[attr-defined]
    scan_daily_records = _timeline_module.scan_daily_records  # type: ignore[attr-defined]
    summarise_project_timeline = _timeline_module.summarise_project_timeline  # type: ignore[attr-defined]
    write_timeline_csv = _timeline_module.write_timeline_csv  # type: ignore[attr-defined]
else:  # pragma: no cover
    from .cli.build_timeline_cli import main, parse_args
    from .core.timeline import (
        build_timeline,
        scan_daily_records,
        summarise_project_timeline,
        write_timeline_csv,
    )

__all__ = [
    "main",
    "parse_args",
    "build_timeline",
    "scan_daily_records",
    "summarise_project_timeline",
    "write_timeline_csv",
]


if __name__ == "__main__":
    main()
