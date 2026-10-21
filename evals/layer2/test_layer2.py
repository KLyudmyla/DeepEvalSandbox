"""Pytest entry point for the real Layer 2 retrieval evaluation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from evals.datasets.golden_dataset import GOLDEN_DATASET
from evals.layer2.__main__ import _build_agent
from evals.layer2.run_layer2 import run_layer2_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def layer2_api_key() -> str:
    """Load the real-agent API key or skip the live evaluation."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is required for the Layer 2 evaluation")
    return api_key


@pytest.fixture(scope="session")
def layer2_agent(layer2_api_key):
    """Build a fresh agent and load the repository knowledge base."""
    return _build_agent(layer2_api_key)


@pytest.mark.parametrize(
    "golden_case",
    GOLDEN_DATASET,
    ids=lambda case: str(case["id"]),
)
def test_layer2_retrieval(layer2_agent, golden_case, tmp_path: Path):
    """Run one golden case so pytest reports its individual status."""
    report_path = tmp_path / f"{golden_case['id']}.json"
    report = run_layer2_evaluation(
        layer2_agent,
        [golden_case],
        report_path=report_path,
        show_progress=True,
    )

    assert report["passed"] is True
    assert report_path.exists()
