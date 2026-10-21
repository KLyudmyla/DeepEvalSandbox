"""Layer 2 result aggregation and threshold assertions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .deepeval_compat import LLMTestCase


_LLM_FIELD_ALIASES: tuple[tuple[str, ...], ...] = (
    ("evaluationCost", "evaluation_cost"),
    ("evaluationModel", "evaluation_model"),
)

_EXCLUDED_REPORT_FIELDS = {"strictMode", "verboseLogs"}


def _metric_value(
    metric_data: Any,
    serialized: Mapping[str, Any],
    *names: str,
) -> Any:
    """Read a metric field from serialized data or the original metric object."""

    for name in names:
        if name in serialized and serialized[name] is not None:
            return serialized[name]
    for name in names:
        value = getattr(metric_data, name, None)
        if value is not None:
            return value
    for name in names:
        if name in serialized:
            return serialized[name]
    return None


def _is_llm_judge_metric(metric_data: Any, serialized: Mapping[str, Any]) -> bool:
    """Identify metrics for which DeepEval used an LLM judge.

    DeepEval has represented the judge fields on metric data with both
    camelCase and snake_case names. Some versions expose those fields on all
    metric-data models, so the GEval marker is preferred over field presence.
    """

    metric_type = type(metric_data).__name__.lower()
    metric_name = str(serialized.get("name") or getattr(metric_data, "name", "")).lower()
    if "geval" in metric_type or "llm" in metric_type or "[geval]" in metric_name:
        return True

    # Some DeepEval versions put the judge fields on every metric-data model,
    # so field presence alone is not enough. A populated judge field is still
    # a useful fallback for versions that omit the ``[GEval]`` name suffix.
    return any(
        _metric_value(metric_data, serialized, *aliases) is not None
        for aliases in _LLM_FIELD_ALIASES
    )


def _threshold_for_metric(
    metric_name: str,
    thresholds: Mapping[str, float],
) -> float | None:
    """Resolve a configured threshold, allowing DeepEval's ``[GEval]`` suffix."""

    if metric_name in thresholds:
        return thresholds[metric_name]
    for configured_name, threshold in thresholds.items():
        if metric_name.startswith(f"{configured_name} ["):
            return threshold
    return None


def resolve_metric_threshold(
    metric_name: str,
    *,
    case_thresholds: Mapping[str, float] | None = None,
    global_thresholds: Mapping[str, float] | None = None,
) -> float | None:
    """Resolve a case override before falling back to the global threshold."""

    threshold = _threshold_for_metric(metric_name, case_thresholds or {})
    if threshold is not None:
        return threshold
    return _threshold_for_metric(metric_name, global_thresholds or {})


def _metric_data_to_dict(metric_data: Any) -> dict[str, Any]:
    serialized: Mapping[str, Any]
    if isinstance(metric_data, Mapping):
        serialized = metric_data
    elif hasattr(metric_data, "model_dump"):
        serialized = metric_data.model_dump(by_alias=True)
    elif hasattr(metric_data, "dict"):
        serialized = metric_data.dict()
    else:
        serialized = {}

    # Keep DeepEval's additional metric fields, but explicitly normalize the
    # fields required by the Layer 2 report. DeepEval versions have exposed
    # the LLM fields as both snake_case and camelCase.
    metric = dict(serialized)
    for field_name in _EXCLUDED_REPORT_FIELDS:
        metric.pop(field_name, None)
    is_llm_judge = _is_llm_judge_metric(metric_data, serialized)
    if not is_llm_judge:
        for aliases in _LLM_FIELD_ALIASES:
            for field_name in aliases:
                metric.pop(field_name, None)
    metric.update(
        {
            "name": _metric_value(metric_data, serialized, "name"),
            "reason": _metric_value(metric_data, serialized, "reason"),
            "score": _metric_value(metric_data, serialized, "score"),
            "error": _metric_value(metric_data, serialized, "error"),
            "threshold": _metric_value(metric_data, serialized, "threshold"),
        }
    )
    if is_llm_judge:
        metric.update(
            {
                "evaluationCost": _metric_value(
                    metric_data,
                    serialized,
                    "evaluationCost",
                    "evaluation_cost",
                ),
                "evaluationModel": _metric_value(
                    metric_data,
                    serialized,
                    "evaluationModel",
                    "evaluation_model",
                ),
            }
        )
    return metric


def aggregate_layer2_report(
        evaluation_result: Any,
        test_cases: Sequence[LLMTestCase],
        *,
        thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Convert DeepEval result objects into the existing unified JSON report."""

    test_results = getattr(evaluation_result, "test_results", []) or []
    cases_report: list[dict[str, Any]] = []
    metric_totals: dict[str, dict[str, int]] = {}

    for index, result in enumerate(test_results):
        case = test_cases[index] if index < len(test_cases) else None
        case_meta = getattr(case, "additional_metadata", {}) or {}
        case_thresholds = case_meta.get("thresholds", {}) if isinstance(case_meta, Mapping) else {}

        metric_rows = [
            _metric_data_to_dict(metric)
            for metric in (getattr(result, "metrics_data", None) or [])
        ]

        for row in metric_rows:
            name = str(row.get("name") or "unknown")

            configured_threshold = resolve_metric_threshold(
                name,
                case_thresholds=case_thresholds,
                global_thresholds=thresholds,
            )

            if configured_threshold is not None:
                row["threshold"] = configured_threshold
                score = row.get("score")
                if score is not None and row.get("error") is None:
                    row["success"] = float(score) >= float(configured_threshold)

            totals = metric_totals.setdefault(name, {"passed": 0, "failed": 0})
            totals["passed" if row.get("success") is True else "failed"] += 1

        result_input = getattr(result, "input", None)
        if result_input is None and case is not None:
            result_input = getattr(case, "input", None)
        case_id = getattr(case, "name", None) if case is not None else None
        if case_id is None:
            case_id = getattr(result, "name", None)

        cases_report.append(
            {
                # Keep ``id`` for consumers of the original report while
                # exposing the requested, unambiguous case identifier.
                "case-id": case_id,
                "id": case_id,
                "input": result_input,
                "success": all(row.get("success") is True for row in metric_rows),
                "metrics": metric_rows,
            }
        )

    failed_metrics = [
        {
            "case_id": case["case-id"],
            "metric": metric["name"],
            "score": metric.get("score"),
            "threshold": metric.get("threshold"),
            "reason": metric.get("reason"),
        }
        for case in cases_report
        for metric in case["metrics"]
        if metric.get("success") is not True
    ]

    return {
        "schema_version": "layer2-retrieval-report/v1",
        "passed": not failed_metrics and all(case["success"] for case in cases_report),
        "thresholds": dict(thresholds or {}),
        "metric_totals": metric_totals,
        "failed_metrics": failed_metrics,
        "cases": cases_report,
        "deepeval": {
            "confident_link": getattr(evaluation_result, "confident_link", None),
            "test_run_id": getattr(evaluation_result, "test_run_id", None),
        },
    }


def assert_layer2_report_passes(report: Mapping[str, Any]) -> None:
    """Raise one actionable assertion for any failed threshold or case."""

    failures = list(report.get("failed_metrics", []))
    if not report.get("passed", False):
        details = "; ".join(
            f"{item.get('case_id')} / {item.get('metric')} "
            f"score={item.get('score')} threshold={item.get('threshold')}"
            for item in failures
        )
        raise AssertionError(
            f"Layer 2 retrieval evaluation failed: "
            f"{details or 'one or more test cases failed'}"
        )


__all__ = [
    "aggregate_layer2_report",
    "assert_layer2_report_passes",
    "resolve_metric_threshold",
]
