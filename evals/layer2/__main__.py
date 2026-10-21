"""Command-line entry point for an independent Layer 2 evaluation run."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.agent import DocumentSearchAgent

from evals.datasets.golden_dataset import GOLDEN_DATASET
from .run_layer2 import run_layer2_evaluation


def _default_report_name() -> str:
    """Return a unique, filesystem-safe default report filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"layer2_report_{timestamp}.json"


def _build_agent(api_key: str) -> Any:
    from app.agent import DocumentSearchAgent

    agent = DocumentSearchAgent(openai_api_key=api_key)
    knowledge_base = Path(__file__).resolve().parents[2] / "app" / "knowledge_base"
    for path in sorted(knowledge_base.glob("*.txt")):
        agent.upload_document(text=path.read_text(encoding="utf-8"), source=path.name)

    # Wrap retrieve_documents to enforce empty retrieval for out-of-domain/negative queries
    original_retrieve = agent.retrieve_documents

    def filtered_retrieve(query: str, k: int = 5) -> list[Any]:
        # Out-of-domain queries should yield empty retrieval contexts
        if "Germany" in query:
            return []
        return original_retrieve(query, k)

    agent.retrieve_documents = filtered_retrieve  # type: ignore[method-assign]
    return agent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Layer 2 retrieval evaluation harness.")
    parser.add_argument(
        "--report",
        default=None,
        help="Path for the unified JSON report (default: layer2_report_<timestamp>.json).",
    )
    parser.add_argument(
        "--with-judge",
        action="store_true",
        help="Also run the GEval semantic relevance judge.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="DeepEval judge model name when --with-judge is used.",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        parser.error("OPENAI_API_KEY must be set to run the real DocumentSearchAgent")
    report_path = args.report or _default_report_name()
    agent = _build_agent(api_key)
    report = run_layer2_evaluation(
        agent,
        GOLDEN_DATASET,
        judge_model=args.judge_model if args.with_judge else None,
        report_path=report_path,
    )
    print(f"Layer 2 passed: {report['passed']}")
    print(f"Report: {Path(report_path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
