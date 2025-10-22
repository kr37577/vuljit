"""Core utilities for RQ3 simulations."""

from . import baseline, io, metrics, predictions, scheduling, simulation, timeline
from .baseline import baseline_detection_metrics, build_threshold_map
from .io import (
    DEFAULTS,
    ensure_directory,
    ensure_parent_directory,
    load_build_counts,
    load_detection_table,
    normalise_path,
    parse_build_counts_csv,
    resolve_default,
    resolve_default_path,
)
from .metrics import (
    aggregate_strategy_metrics,
    prepare_daily_totals,
    prepare_project_metrics,
    safe_ratio,
    summarize_schedule,
    summarize_schedule_by_project,
)
from .predictions import (
    collect_predictions,
    iter_prediction_files,
    load_project_predictions,
)
from .scheduling import (
    STRATEGY_REGISTRY,
    get_strategy,
    iter_strategies,
    list_strategies,
    normalize_name,
    run_strategy,
)
from .simulation import (
    SimulationResult,
    normalize_to_date,
    prepare_schedule_for_waste_analysis,
    run_minimal_simulation,
    summarize_wasted_builds,
)
from .timeline import (
    build_timeline,
    scan_daily_records,
    summarise_project_timeline,
    write_timeline_csv,
)
from .plotting import plot_additional_builds_boxplot

__all__ = [
    "baseline",
    "metrics",
    "io",
    "predictions",
    "scheduling",
    "simulation",
    "timeline",
    "DEFAULTS",
    "ensure_directory",
    "ensure_parent_directory",
    "load_build_counts",
    "load_detection_table",
    "normalise_path",
    "parse_build_counts_csv",
    "resolve_default",
    "resolve_default_path",
    "baseline_detection_metrics",
    "build_threshold_map",
    "safe_ratio",
    "summarize_schedule",
    "summarize_schedule_by_project",
    "prepare_project_metrics",
    "aggregate_strategy_metrics",
    "prepare_daily_totals",
    "iter_prediction_files",
    "load_project_predictions",
    "collect_predictions",
    "STRATEGY_REGISTRY",
    "normalize_name",
    "get_strategy",
    "run_strategy",
    "iter_strategies",
    "list_strategies",
    "SimulationResult",
    "normalize_to_date",
    "prepare_schedule_for_waste_analysis",
    "run_minimal_simulation",
    "summarize_wasted_builds",
    "build_timeline",
    "scan_daily_records",
    "summarise_project_timeline",
    "write_timeline_csv",
    "plot_additional_builds_boxplot",
]
