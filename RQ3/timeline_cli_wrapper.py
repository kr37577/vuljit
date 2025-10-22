"""Backward-compatible wrapper for the build timeline CLI."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _wrapper_dir = Path(__file__).resolve().parent
    _repo_root = _wrapper_dir.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from RQ3.cli.build_timeline_cli import main, parse_args  # type: ignore
    from RQ3.core.timeline import (  # type: ignore
        build_timeline,
        scan_daily_records,
        summarise_project_timeline,
        write_timeline_csv,
    )
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
