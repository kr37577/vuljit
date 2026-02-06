#!/usr/bin/env python3
"""Backward-compatible wrapper for the minimal additional-build simulation CLI."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _wrapper_dir = Path(__file__).resolve().parent
    _analysis_root = _wrapper_dir.parent
    _repo_root = _analysis_root.parent
    _project_root = _repo_root.parent
    for path in (_analysis_root, _repo_root, _project_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

try:
    from .cli.minimal_simulation_cli import (
        load_detection_baseline,
        main,
    )
except ImportError:  # pragma: no cover
    try:
        from analysis.research_question3.cli.minimal_simulation_cli import (
            load_detection_baseline,
            main,
        )
    except ImportError:
        from RQ3.cli.minimal_simulation_cli import (  # type: ignore
            load_detection_baseline,
            main,
        )

try:
    from .core.simulation import run_minimal_simulation
except ImportError:  # pragma: no cover
    try:
        from analysis.research_question3.core.simulation import run_minimal_simulation
    except ImportError:
        from RQ3.core.simulation import run_minimal_simulation  # type: ignore

__all__ = ["main", "load_detection_baseline", "run_minimal_simulation"]


if __name__ == "__main__":
    main()
