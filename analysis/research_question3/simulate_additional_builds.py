#!/usr/bin/env python3
"""Backward-compatible wrapper for the additional-build simulation CLI."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def _import_cli_module(package_name: str) -> ModuleType:
    """Import the additional-builds CLI module from ``package_name``."""

    return importlib.import_module(f"{package_name}.cli.additional_builds_cli")


if __package__ in (None, ""):
    _wrapper_dir = Path(__file__).resolve().parent
    _analysis_root = _wrapper_dir.parent
    _repo_root = _analysis_root.parent
    _project_root = _repo_root.parent

    for path in (_analysis_root, _repo_root, _project_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        _cli_module = _import_cli_module("RQ3")
    except ModuleNotFoundError:
        try:
            _cli_module = _import_cli_module("analysis.research_question3")
        except ModuleNotFoundError:
            _cli_module = _import_cli_module("vuljit.analysis.research_question3")

    main = _cli_module.main  # type: ignore[attr-defined]
else:  # pragma: no cover
    from .cli.additional_builds_cli import main


if __name__ == "__main__":
    main()
