#!/usr/bin/env python3
"""Backward-compatible wrapper for the additional-build simulation CLI."""

from __future__ import annotations

try:
    from .cli.additional_builds_cli import main
except ImportError:  # pragma: no cover
    from RQ3.cli.additional_builds_cli import main  # type: ignore


if __name__ == "__main__":
    main()
