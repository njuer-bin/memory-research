"""Deprecated shim.

Attribution now lives in :mod:`evaluation.attribution` (single source of
truth). This module previously contained a second, divergent implementation
whose E0 gate used the model's failed answer as the judge candidate
(circular: E0 depended on the failure being attributed). See
docs/AUDIT-2026-10-18.md finding C2.
"""
from evaluation.attribution import (  # noqa: F401
    ATTRIBUTION_STAGES,
    STAGE_TO_FAILURE,
    FailureLabel,
    attribute_failure,
    control_failure_label,
    earliest_failed_stage,
)
