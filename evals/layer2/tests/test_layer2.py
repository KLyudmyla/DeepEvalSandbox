"""Opt-in pytest entry point for the real Layer 2 retrieval evaluation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evals.datasets.golden_dataset_l2 import GOLDEN_DATASET
from evals.layer2.__main__ import _build_agent
from evals.layer2.run_layer2 import run_layer2_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.live


@pytest.fixture(scope="session")
def layer2_api_key() -> str:
    """Load the real-agent API key or skip the live evaluation."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is required for the Layer 2 evaluation")
    return api_key


@pytest.fixture(scope="session")
def layer2_agent(layer2_api_key):
    """Build the real agent and load/reuse the validated knowledge-base index."""
    return _build_agent(layer2_api_key)


@pytest.fixture(scope="session")
def layer2_report(layer2_agent, request, tmp_path_factory: pytest.TempPathFactory):
    """Run all cases once and retain complete per-case results."""
    report_path = tmp_path_factory.mktemp("layer2-live") / "layer2.json"
    judge_model = (
        None
        if request.config.getoption("--layer2-without-judge")
        else request.config.getoption("--layer2-judge-model")
    )
    report = run_layer2_evaluation(
        layer2_agent,
        GOLDEN_DATASET,
        judge_model=judge_model,
        report_path=report_path,
        fail_on_threshold=False,
        max_judge_cost_usd=request.config.getoption("--layer2-max-judge-cost-usd"),
        show_progress=False,
    )
    expected_ids = {str(case["id"]) for case in GOLDEN_DATASET}
    actual_ids = {str(case["case-id"]) for case in report.get("cases", [])}
    assert actual_ids == expected_ids, (
        f"Live report case coverage mismatch: expected {len(expected_ids)}, "
        f"got {len(actual_ids)}"
    )
    assert report_path.exists()
    return report


@pytest.mark.parametrize(
    "golden_case",
    GOLDEN_DATASET,
    ids=lambda case: str(case["id"]),
)
def test_layer2_retrieval(layer2_report, golden_case):
    """Expose each live case as an individual pytest result after one run."""
    case_id = str(golden_case["id"])
    case_report = next(case for case in layer2_report["cases"] if case["case-id"] == case_id)
    assert case_report["success"] is True, (
        f"{case_id} failed: "
        f"{[metric for metric in case_report['metrics'] if metric.get('success') is False]!r}"
    )


def test_layer2_report_passes_all_gates(layer2_report):
    """Apply run-level budget and report gates without hiding per-case results."""
    assert layer2_report["passed"] is True, layer2_report["failed_metrics"]
