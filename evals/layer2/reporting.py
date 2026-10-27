"""Layer 2 result aggregation and threshold assertions."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .deepeval_bridge import LLMTestCase


_LLM_FIELD_ALIASES: tuple[tuple[str, ...], ...] = (
    ("evaluationCost", "evaluation_cost"),
    ("evaluationModel", "evaluation_model"),
)

_EXCLUDED_REPORT_FIELDS = {"strictMode", "verboseLogs"}
_RANKING_METRICS = {"Recall@k", "Precision@k", "Hit Rate@k", "MRR", "nDCG@k"}
_SEMANTIC_RELEVANCE_METRIC = ("Semantic document relevance (0-3)")
_EMPTY_NEGATIVE_NOT_APPLICABLE_METRICS = {"Source document ID provenance"}


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


def _is_judge_row(row: Mapping[str, Any]) -> bool:
    name = str(row.get("name") or "").lower()
    return (
        "[geval]" in name
        or row.get("evaluationModel") is not None
        or row.get("evaluation_model") is not None
    )


def _known_cost(value: Any) -> float | None:
    if value is None:
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


def _cost_summary(
    cases_report: Sequence[Mapping[str, Any]],
    *,
    judge_enabled: bool,
    index_info: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Summarize known costs without treating missing telemetry as free."""
    judge_rows = [
        metric
        for case in cases_report
        for metric in case.get("metrics", [])
        if (
                isinstance(metric, Mapping)
                and _is_judge_row(metric)
                and metric.get("applicable") is not False
        )
    ]
    known_costs = [
        cost
        for metric in judge_rows
        if (cost := _known_cost(metric.get("evaluationCost", metric.get("evaluation_cost")))) is not None
    ]
    unknown_count = len(judge_rows) - len(known_costs)

    if not judge_enabled:
        judge = {
            "enabled": False,
            "total_usd": 0.0,
            "known_usd": 0.0,
            "complete": True,
            "unknown_metric_count": 0,
            "source": "no GEval judge was enabled",
        }
    else:
        known_total = sum(known_costs)
        complete = bool(judge_rows) and unknown_count == 0
        judge = {
            "enabled": True,
            "total_usd": known_total if complete else None,
            "known_usd": known_total if known_costs else None,
            "complete": complete,
            "unknown_metric_count": unknown_count if judge_rows else 1,
            "source": ("DeepEval MetricData.evaluationCost; ""per-case GEval cost with metric cost reset before each "
                       "evaluation"),
        }

    embedding = {
        "total_usd": None,
        "known_usd": None,
        "complete": False,
        "source": "OpenAI embedding usage is not exposed by DocumentSearchAgent/Chroma",
    }
    if index_info:
        embedding.update(
            {
                "index_cache_hit": index_info.get("cache_hit"),
                "index_fingerprint": index_info.get("fingerprint"),
                "corpus_documents": index_info.get("corpus_documents"),
            }
        )

    return {
        "judge": judge,
        "embeddings": embedding,
        "total_usd": None,
        "known_total_usd": judge["known_usd"],
    }

def _is_semantic_relevance_metric(metric_name: str) -> bool:
    """Recognize semantic relevance with or without the GEval suffix."""

    return (
        metric_name == _SEMANTIC_RELEVANCE_METRIC
        or metric_name.startswith(
            f"{_SEMANTIC_RELEVANCE_METRIC} ["
        )
    )

def aggregate_layer2_report(
    evaluation_result: Any,
    test_cases: Sequence[LLMTestCase],
    *,
    thresholds: Mapping[str, float] | None = None,
    expected_metrics: Sequence[str] | None = None,
    judge_enabled: bool | None = None,
    max_judge_cost_usd: float | None = None,
    index_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert DeepEval results into a unified Layer 2 report."""

    test_results = (
        getattr(evaluation_result, "test_results", []) or []
    )

    cases_report: list[dict[str, Any]] = []

    metric_totals: dict[str, dict[str, int]] = {}

    expected_metric_names = list(expected_metrics or [])

    cases_by_name = {
        str(case.name): case
        for case in test_cases
    }

    if len(cases_by_name) != len(test_cases):
        raise ValueError("Duplicate input case names")

    seen_names: set[str] = set()

    # ---------------------------------------------------------
    # 1. Process individual cases
    # ---------------------------------------------------------

    for index, result in enumerate(test_results):

        result_name = getattr(result, "name", None)

        case = (
            cases_by_name.get(str(result_name))
            if result_name is not None
            else (
                test_cases[index]
                if index < len(test_cases)
                else None
            )
        )

        if case is None:
            raise ValueError(
                f"DeepEval returned an unknown or unnamed case: "
                f"{result_name!r}"
            )

        if str(case.name) in seen_names:
            raise ValueError(
                f"DeepEval returned a duplicate case: "
                f"{result_name!r}"
            )

        seen_names.add(str(case.name))

        case_meta = (
            getattr(case, "additional_metadata", {}) or {}
        )

        case_thresholds = (
            case_meta.get("thresholds", {})
            if isinstance(case_meta, Mapping)
            else {}
        )

        query_type = case_meta.get("query_type")

        if query_type not in {"answerable", "unanswerable"}:
            raise ValueError(
                f"{case.name}: invalid query_type={query_type!r}"
            )

        negative_case = query_type == "unanswerable"

        # -----------------------------------------------------
        # 2. Normalize metric rows
        # -----------------------------------------------------

        metric_rows = [
            _metric_data_to_dict(metric)
            for metric in (
                getattr(result, "metrics_data", None) or []
            )
        ]

        metric_names = [
            str(row.get("name") or "")
            for row in metric_rows
        ]

        if len(metric_names) != len(set(metric_names)):
            raise ValueError(
                f"Duplicate metric rows for case {case.name!r}"
            )

        # -----------------------------------------------------
        # 3. Validate GEval applicability
        # -----------------------------------------------------

        if negative_case:

            unexpected_geval = [
                row
                for row in metric_rows
                if _is_semantic_relevance_metric(
                    str(row.get("name") or "")
                )
            ]

            if unexpected_geval:
                raise ValueError(
                    f"{case.name}: GEval must not execute "
                    f"for unanswerable queries"
                )

        # -----------------------------------------------------
        # 4. Evaluate metric results
        # -----------------------------------------------------

        for row in metric_rows:

            name = str(row.get("name") or "unknown")

            empty_negative_case = (
                negative_case
                and not case_meta.get(
                    "retrieved_document_ids",
                    [],
                )
            )

            not_applicable = (
                negative_case
                and name in _RANKING_METRICS
            ) or (
                empty_negative_case
                and name in (
                    _EMPTY_NEGATIVE_NOT_APPLICABLE_METRICS
                )
            )

            if not_applicable:

                row["applicable"] = False
                row["success"] = None

                continue

            row["applicable"] = True

            configured_threshold = resolve_metric_threshold(
                name,
                case_thresholds=case_thresholds,
                global_thresholds=thresholds,
            )

            if configured_threshold is not None:
                row["threshold"] = configured_threshold

            score = row.get("score")

            if (
                score is not None
                and row.get("error") is None
            ):

                numeric = float(score)

                row["success"] = (
                    math.isfinite(numeric)
                    and configured_threshold is not None
                    and numeric >= float(configured_threshold)
                )

            else:
                row["success"] = False

            totals = metric_totals.setdefault(
                name,
                {
                    "passed": 0,
                    "failed": 0,
                    "not_applicable": 0,
                },
            )

            if row["success"] is True:
                totals["passed"] += 1
            else:
                totals["failed"] += 1

        # -----------------------------------------------------
        # 5. Add GEval N/A for unanswerable queries
        # -----------------------------------------------------

        if negative_case and judge_enabled:

            metric_rows.append(
                {
                    "name": (
                        "Semantic document relevance "
                        "(0-3) [GEval]"
                    ),
                    "applicable": False,
                    "score": None,
                    "success": None,
                    "threshold": thresholds.get(
                        _SEMANTIC_RELEVANCE_METRIC
                    ) if thresholds else None,
                    "reason": (
                        "Not applicable: unanswerable query"
                    ),
                    "evaluationCost": None,
                    "evaluationModel": None,
                }
            )

            totals = metric_totals.setdefault(
                "Semantic document relevance (0-3) [GEval]",
                {
                    "passed": 0,
                    "failed": 0,
                    "not_applicable": 0,
                },
            )

            totals["not_applicable"] += 1

        # -----------------------------------------------------
        # 6. Validate required metrics
        # -----------------------------------------------------

        missing_metrics = []

        for expected in expected_metric_names:

            # GEval is intentionally absent for unanswerable
            # queries and represented by an N/A row.
            if (
                negative_case
                and _is_semantic_relevance_metric(expected)
            ):
                continue

            found = any(
                str(row.get("name") or "") == expected
                or str(row.get("name") or "").startswith(
                    f"{expected} ["
                )
                for row in metric_rows
            )

            if not found:
                missing_metrics.append(expected)

        # -----------------------------------------------------
        # 7. Build case report
        # -----------------------------------------------------

        result_input = getattr(result, "input", None)

        if result_input is None:
            result_input = getattr(case, "input", None)

        case_id = getattr(case, "name", None)

        case_success = (
            not missing_metrics
            and any(
                row.get("success") is True
                for row in metric_rows
            )
            and all(
                row.get("success") in (True, None)
                for row in metric_rows
            )
        )

        cases_report.append(
            {
                "case-id": case_id,
                "id": case_id,
                "input": result_input,
                "query_type": query_type,
                "success": case_success,
                "missing_metrics": missing_metrics,
                "metrics": metric_rows,
            }
        )

    # ---------------------------------------------------------
    # 8. Collect failed metrics
    # ---------------------------------------------------------

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
        if metric.get("success") not in (True, None)
    ]

    for case in cases_report:

        for metric_name in case.get(
            "missing_metrics",
            [],
        ):

            failed_metrics.append(
                {
                    "case_id": case["case-id"],
                    "metric": metric_name,
                    "score": None,
                    "threshold": None,
                    "reason": (
                        "DeepEval did not return "
                        "the required metric row"
                    ),
                }
            )

    # ---------------------------------------------------------
    # 9. Calculate costs
    # ---------------------------------------------------------

    costs = _cost_summary(
        cases_report,
        judge_enabled=(
            bool(judge_enabled)
            if judge_enabled is not None
            else any(
                _is_judge_row(metric)
                for case in cases_report
                for metric in case["metrics"]
            )
        ),
        index_info=index_info,
    )

    # ---------------------------------------------------------
    # 10. Enforce judge budget
    # ---------------------------------------------------------

    budget: dict[str, Any] = {
        "max_judge_cost_usd": max_judge_cost_usd,
        "status": (
            "not_configured"
            if max_judge_cost_usd is None
            else "unknown"
        ),
    }

    if max_judge_cost_usd is not None:

        if (
            not math.isfinite(float(max_judge_cost_usd))
            or float(max_judge_cost_usd) < 0
        ):
            raise ValueError(
                "max_judge_cost_usd must be "
                "a finite non-negative number"
            )

        judge_cost = costs["judge"]

        if (
            not judge_cost["enabled"]
            or (
                judge_cost["complete"]
                and judge_cost["total_usd"]
                <= max_judge_cost_usd
            )
        ):
            budget["status"] = "within_limit"

        elif judge_cost["complete"]:

            budget["status"] = "exceeded"

            failed_metrics.append(
                {
                    "case_id": "__evaluation__",
                    "metric": "judge evaluation budget",
                    "score": judge_cost["total_usd"],
                    "threshold": max_judge_cost_usd,
                    "reason": (
                        "Known GEval cost exceeded "
                        "the configured budget"
                    ),
                }
            )

        else:

            budget["status"] = "unknown_cost"

            failed_metrics.append(
                {
                    "case_id": "__evaluation__",
                    "metric": "judge evaluation budget",
                    "score": judge_cost["known_usd"],
                    "threshold": max_judge_cost_usd,
                    "reason": (
                        "DeepEval did not expose "
                        "complete judge cost telemetry"
                    ),
                }
            )

    # ---------------------------------------------------------
    # 11. Final report
    # ---------------------------------------------------------

    return {
        "schema_version": "layer2-retrieval-report/v1",
        "passed": (
            bool(cases_report)
            and len(cases_report) == len(test_cases)
            and not failed_metrics
            and all(
                case["success"]
                for case in cases_report
            )
        ),
        "thresholds": dict(thresholds or {}),
        "metric_totals": metric_totals,
        "failed_metrics": failed_metrics,
        "costs": costs,
        "budget": budget,
        "cases": cases_report,
        "deepeval": {
            "confident_link": getattr(
                evaluation_result,
                "confident_link",
                None,
            ),
            "test_run_id": getattr(
                evaluation_result,
                "test_run_id",
                None,
            ),
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
