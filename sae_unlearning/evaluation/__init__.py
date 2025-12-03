"""Evaluation module for metrics and evaluation functions."""

from .metrics import retention_metric, alignment_metric
from .evaluator import (
    evaluate_multiple_choice,
    evaluate_with_interventions,
    run_baseline_evaluation,
    format_multiple_choice_prompt,
)

__all__ = [
    "retention_metric",
    "alignment_metric",
    "evaluate_multiple_choice",
    "evaluate_with_interventions",
    "run_baseline_evaluation",
    "format_multiple_choice_prompt",
]

