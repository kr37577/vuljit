"""CLI entry point for precision-threshold analysis."""

from __future__ import annotations

from ..threshold_precision_analysis import main as _main, parse_args as _parse_args

__all__ = ["parse_args", "main"]


def parse_args():
    """Proxy to ``threshold_precision_analysis.parse_args``."""

    return _parse_args()


def main() -> None:
    """Run the precision-threshold analysis CLI."""

    _main()
