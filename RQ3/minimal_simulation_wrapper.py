#!/usr/bin/env python3
"""Backward-compatible wrapper for the minimal additional-build simulation CLI."""

from __future__ import annotations

try:
    from .cli.minimal_simulation_cli import (
        load_detection_baseline,
        main,
    )
except ImportError:  # pragma: no cover
    from RQ3.cli.minimal_simulation_cli import (  # type: ignore
        load_detection_baseline,
        main,
    )

try:
    from .core.simulation import run_minimal_simulation
except ImportError:  # pragma: no cover
    from RQ3.core.simulation import run_minimal_simulation  # type: ignore

__all__ = ["main", "load_detection_baseline", "run_minimal_simulation"]


if __name__ == "__main__":
    main()
