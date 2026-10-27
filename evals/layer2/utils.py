"""Layer 2 retrieval normalization, validation, and DeepEval test-case builders."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from evals.datasets.golden_dataset_l2 import GOLDEN_DATASET
from jsonschema import Draft202012Validator

from .config import DEFAULT_K, GOLDEN_DATASET_SCHEMA, SEMANTIC_RELEVANCE_MAX
from .deepeval_bridge import LLMTestCase


@dataclass(frozen=True)
class RetrievedDocument:
    """Normalized retrieval output used by the adapter and report builder."""

    document_id: str
    content: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    rank_score: float | None = None


def document_id(value: Any) -> str:
    """Extract and normalize a non-empty document ID."""

    if isinstance(value, RetrievedDocument):
        candidate = value.document_id
    elif isinstance(value, Mapping):
        metadata = value.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        candidate = value.get("document_id") or metadata.get("source") or value.get("source") or value.get("id")
    else:
        candidate = getattr(value, "document_id", None)
        if candidate is None:
            metadata = getattr(value, "metadata", None)
            metadata = metadata if isinstance(metadata, Mapping) else {}
            candidate = metadata.get("source") or getattr(value, "source", None)
        if candidate is None and isinstance(value, str):
            candidate = value
    if candidate is None:
        raise ValueError(f"Unable to extract a document ID from {value!r}")
    result = str(candidate).strip()
    if not result:
        raise ValueError("Document IDs must be non-empty strings")
    return result


def stable_unique_ids(values: Iterable[Any]) -> list[str]:
    """Return IDs in ranked order, retaining only the first occurrence."""

    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        identifier = document_id(value)
        if identifier not in seen:
            seen.add(identifier)
            unique.append(identifier)
    return unique


def validate_k(k: int | None) -> int:
    if k is None:
        return 0
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer or None")
    return k


def positive_relevance_map(graded_relevance: Mapping[Any, Any] | None) -> dict[str, int]:
    """Normalize graded relevance and retain only positive grades."""

    result: dict[str, int] = {}
    for identifier, raw_score in (graded_relevance or {}).items():
        doc_id = document_id(identifier)
        if isinstance(raw_score, bool) or not isinstance(raw_score, int):
            raise ValueError(f"Relevance for {doc_id!r} must be an integer from 0 to 3")
        if raw_score < 0 or raw_score > SEMANTIC_RELEVANCE_MAX:
            raise ValueError(f"Relevance for {doc_id!r} must be between 0 and 3")
        result[doc_id] = max(result.get(doc_id, 0), raw_score)
    return {doc_id: score for doc_id, score in result.items() if score > 0}


def normalize_retrieved_documents(results: Iterable[Any]) -> list[RetrievedDocument]:
    """Normalize retrieval output while preserving retriever order and rank slots."""

    normalized: list[tuple[int, RetrievedDocument]] = []
    for position, result in enumerate(results or []):
        if isinstance(result, RetrievedDocument):
            document = result
        elif isinstance(result, Mapping):
            metadata = result.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
            raw_score = result.get("rank_score", result.get("score"))
            document = RetrievedDocument(
                document_id=document_id(result),
                content=str(result.get("content", result.get("page_content", "")) or ""),
                metadata=metadata,
                rank_score=float(raw_score) if raw_score is not None else None,
            )
        else:
            metadata = getattr(result, "metadata", {})
            raw_score = getattr(result, "rank_score", getattr(result, "score", None))
            document = RetrievedDocument(
                document_id=document_id(result),
                content=str(getattr(result, "page_content", getattr(result, "content", "")) or ""),
                metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
                rank_score=float(raw_score) if raw_score is not None else None,
            )
        normalized.append((position, document))


    return [document for _, document in normalized]


def exact_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON-like values without permitting numeric/string coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(exact_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact_json_equal(a, b) for a, b in zip(left, right))
    return left == right


def validate_golden_dataset(dataset: Sequence[Mapping[str, Any]]) -> None:
    """Fail fast on malformed or internally inconsistent golden data."""

    schema_errors = sorted(
        Draft202012Validator(GOLDEN_DATASET_SCHEMA).iter_errors(dataset),
        key=lambda error: list(error.absolute_path),
    )
    if schema_errors:
        error = schema_errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<dataset>"
        raise ValueError(f"Golden dataset schema error at {location}: {error.message}")

    seen_case_ids: set[str] = set()
    for case in dataset:
        case_id = case["id"].strip()
        query = case["query"].strip()
        if not case_id or not query or case_id in seen_case_ids:
            raise ValueError(f"Golden case ID/query is blank or duplicated: {case_id!r}")
        seen_case_ids.add(case_id)

        raw_ground_truth = case["ground_truth_document_ids"]
        if isinstance(raw_ground_truth, (str, bytes)) or not isinstance(raw_ground_truth, Sequence):
            raise ValueError(f"{case_id}: ground_truth_document_ids must be an array")
        ground_truth = [document_id(value) for value in raw_ground_truth]
        expected_query_type = (
            "answerable" if ground_truth else "unanswerable"
        )

        if case["query_type"] != expected_query_type:
            raise ValueError(
                f"{case_id}: query_type={case['query_type']!r} "
                f"is inconsistent with ground_truth_document_ids. "
                f"Expected {expected_query_type!r}."
            )
        if len(ground_truth) != len(set(ground_truth)):
            raise ValueError(f"{case_id}: ground_truth_document_ids must not contain duplicates")

        raw_scores = case["graded_relevance_scores"]
        score_ids = [document_id(identifier) for identifier in raw_scores]
        if len(score_ids) != len(set(score_ids)):
            raise ValueError(f"{case_id}: graded_relevance_scores must not contain duplicate document IDs")
        scores = positive_relevance_map(raw_scores)
        # Weakly relevant distractors may be graded without being binary ground truth.
        if any(scores.get(doc_id, 0) < 2 for doc_id in ground_truth):
            raise ValueError(f"{case_id}: ground-truth IDs must have relevance grade 2 or 3")
        validate_k(case.get("k", DEFAULT_K))
        if len(ground_truth) > case.get("k", DEFAULT_K):
            raise ValueError(f"{case_id}: k cannot cover all ground-truth documents")

def _to_retriever(agent_or_retriever: Any) -> Callable[[str, int], Any]:
    if callable(agent_or_retriever):
        return agent_or_retriever
    retrieve = getattr(agent_or_retriever, "retrieve_documents", None)
    if not callable(retrieve):
        raise TypeError("Provide a callable retriever or an object with retrieve_documents(query, k)")
    return retrieve


def build_layer2_test_cases(
    agent_or_retriever: Any,
    dataset: Sequence[Mapping[str, Any]] = GOLDEN_DATASET,
) -> list[LLMTestCase]:
    """Run retrieval and convert every golden case into an LLMTestCase."""

    validate_golden_dataset(dataset)

    retrieve = _to_retriever(agent_or_retriever)
    cases: list[LLMTestCase] = []

    for golden in dataset:
        k = int(golden.get("k", DEFAULT_K))

        result = retrieve(golden["query"], k)

        if inspect.isawaitable(result):
            result = asyncio.run(result)

        documents = normalize_retrieved_documents(result or [])[:k]

        ground_truth_ids = list(
            golden["ground_truth_document_ids"]
        )

        query_type = (
            "answerable"
            if ground_truth_ids
            else "unanswerable"
        )

        metadata = {
            "golden_id": golden["id"],
            "query_type": query_type,
            "retrieved_document_ids": [
                document.document_id
                for document in documents
            ],
            "retrieved_metadata": [
                dict(document.metadata)
                for document in documents
            ],
            "expected_metadata": golden.get("expected_metadata"),
            "ground_truth_document_ids": ground_truth_ids,
            "expected_metadata_filters": dict(
                golden["expected_metadata_filters"]
            ),
            "graded_relevance_scores": dict(
                golden["graded_relevance_scores"]
            ),
            "k": k,
            "thresholds": dict(golden.get("thresholds", {})),
        }

        expected_output = json.dumps(
            {
                "ground_truth_document_ids": ground_truth_ids,
                "graded_relevance_scores": golden[
                    "graded_relevance_scores"
                ],
            },
            sort_keys=True,
        )

        cases.append(
            LLMTestCase(
                name=str(golden["id"]),
                input=str(golden["query"]),
                actual_output="",
                expected_output=expected_output,
                retrieval_context=[
                    document.content
                    for document in documents
                ],
                additional_metadata=metadata,
                tags=list(golden.get("tags", [])),
            )
        )

    return cases


__all__ = [
    "RetrievedDocument",
    "document_id",
    "stable_unique_ids",
    "validate_k",
    "positive_relevance_map",
    "normalize_retrieved_documents",
    "exact_json_equal",
    "validate_golden_dataset",
    "build_layer2_test_cases",
]
