"""Deterministic retrieval metrics and DeepEval metric implementations.
Metrics like Recall@k, Precision@k, Hit Rate@k, MRR, nDCG@k, Source document ID provenance, and Metadata key-value
fidelity are calculated entirely locally using Python code without making calls to external AI models"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any, Mapping, Sequence

from .config import DEFAULT_K, SEMANTIC_RELEVANCE_MAX
from .deepeval_bridge import BaseMetric, GEval, LLMTestCase, LLMTestCaseParams, Rubric
from .utils import (
    RetrievedDocument,
    document_id,
    exact_json_equal,
    positive_relevance_map,
    stable_unique_ids,
    validate_k,
)


def recall_at_k(
    retrieved_document_ids: Sequence[Any],
    ground_truth_document_ids: Sequence[Any],
    k: int,
) -> float:
    """Recall@k over unique IDs; empty ground truth is not scored for retrieval ranking."""

    k = validate_k(k)
    relevant = set(stable_unique_ids(ground_truth_document_ids))
    retrieved = set([document_id(value) for value in retrieved_document_ids[:k]])

    if not relevant:
        return 1.0  # Not applicable for unsupported queries; report separately.

    if k == 0:
        return 0.0
    return len(relevant & retrieved) / len(relevant)


def precision_at_k(
    retrieved_document_ids: Sequence[Any],
    ground_truth_document_ids: Sequence[Any],
    k: int,
) -> float:
    """Precision@k with a fixed denominator of k; empty ground truth is not scored for retrieval ranking."""

    k = validate_k(k)
    if k == 0:
        return 0.0
    relevant = set(stable_unique_ids(ground_truth_document_ids))
    retrieved = set([document_id(value) for value in retrieved_document_ids[:k]])

    if not relevant:
        return 1.0  # Not applicable for unsupported queries; report separately.

    return len(relevant & retrieved) / k


def hit_rate_at_k(
    retrieved_document_ids: Sequence[Any],
    ground_truth_document_ids: Sequence[Any],
    k: int,
) -> float:
    """Return 1.0 for unlabeled negative cases, otherwise test for a relevant hit."""

    k = validate_k(k)
    if k == 0:
        return 0.0
    relevant = set(stable_unique_ids(ground_truth_document_ids))
    retrieved = set([document_id(value) for value in retrieved_document_ids[:k]])

    if not relevant:
        return 1.0  # Not applicable for unsupported queries; report separately.

    return float(bool(relevant & retrieved))


def reciprocal_rank(
    retrieved_document_ids: Sequence[Any],
    ground_truth_document_ids: Sequence[Any],
    k: int | None = None,
) -> float:
    """Reciprocal rank of first unique relevant result; empty ground truth is not scored for retrieval ranking."""

    if k is not None:
        k = validate_k(k)
    if k == 0:
        return 0.0
    relevant = set(stable_unique_ids(ground_truth_document_ids))
    ranked = [document_id(value) for value in retrieved_document_ids]
    if k is not None:
        ranked = ranked[:k]

    if not relevant:
        return 1.0  # Not applicable for unsupported queries; report separately.

    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _discounted_gain(relevance: Sequence[int]) -> float:
    return sum(
        (2**int(score) - 1) / math.log2(rank + 2)
        for rank, score in enumerate(relevance)
    )


def ndcg_at_k(
    retrieved_document_ids: Sequence[Any],
    graded_relevance_scores: Mapping[Any, Any],
    k: int,
) -> float:
    """nDCG@k; empty ground truth/relevance is not scored for retrieval ranking."""

    k = validate_k(k)
    if k == 0:
        return 0.0
    relevance = positive_relevance_map(graded_relevance_scores)
    # nDCG is evaluated over ranked unique documents. Duplicate handling is
    # reported separately by duplicate_free_rate_at_k so nDCG remains a
    # standard ranking metric while duplicate retrieval is still a hard gate.
    ranked_ids = stable_unique_ids(retrieved_document_ids[:k])

    if not relevance:
        return 1.0  # Not applicable for unsupported queries; report separately.

    actual = [relevance.get(doc_id, 0) for doc_id in ranked_ids]
    ideal_ids = sorted(relevance, key=lambda doc_id: (-relevance[doc_id], doc_id))[:k]
    ideal_gain = _discounted_gain([relevance[doc_id] for doc_id in ideal_ids])
    return 0.0 if ideal_gain == 0 else _discounted_gain(actual) / ideal_gain


def duplicate_free_rate_at_k(
    retrieved_document_ids: Sequence[Any],
    k: int,
) -> float:
    """Return the fraction of observed top-k slots that are first occurrences.

    Empty retrieval and a retrieval shorter than ``k`` are not penalized for
    missing results; ranking coverage metrics handle that separately.
    """

    k = validate_k(k)
    ranked_ids = [document_id(value) for value in retrieved_document_ids[:k]]
    if not ranked_ids:
        return 1.0
    return len(stable_unique_ids(ranked_ids)) / len(ranked_ids)


class _DeterministicMetric(BaseMetric, ABC):
    """BaseMetric adapter for pure, deterministic assertions."""

    def __init__(self, name: str, threshold: float = 1.0) -> None:
        self.name = name
        self.threshold = float(threshold)
        self.strict_mode = True
        self.async_mode = False
        self.verbose_mode = False
        self.include_reason = True
        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool | None = None
        self.applicable: bool = True
        self.error: str | None = None

    @property
    def __name__(self) -> str:
        return self.name

    def is_successful(self) -> bool:
        self.success = self.error is None and self.score is not None and self.score >= self.threshold
        return bool(self.success)

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case, *args, **kwargs)

    @abstractmethod
    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        pass

    @staticmethod
    def _metadata(test_case: LLMTestCase) -> dict[str, Any]:
        metadata = getattr(test_case, "additional_metadata", None) or {}
        if not isinstance(metadata, MutableMapping):
            raise TypeError("LLMTestCase.additional_metadata must be a mapping")
        return dict(metadata)


class RetrievalMetric(_DeterministicMetric):
    """Base class for deterministic Recall/Precision/Hit/MRR/nDCG metrics."""

    def _inputs(self, test_case: LLMTestCase) -> tuple[list[str], list[str], int, dict[str, int]]:
        data = self._metadata(test_case)
        return (
            list(data.get("retrieved_document_ids", [])),
            stable_unique_ids(data.get("ground_truth_document_ids", [])),
            validate_k(data.get("k", DEFAULT_K)),
            dict(data.get("graded_relevance_scores", {})),
        )

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        retrieved, expected, k, graded = self._inputs(test_case)
        self.score = float(self._calculate(retrieved, expected, graded, k))
        self.reason = f"{self.name}={self.score:.6f}; retrieved={retrieved[:k]!r}; expected={expected!r}"
        self.success = self.is_successful()
        return self.score

    @abstractmethod
    def _calculate(
        self,
        retrieved: list[str],
        expected: list[str],
        graded: dict[str, int],
        k: int,
    ) -> float:
        pass


class RecallAtKMetric(RetrievalMetric):
    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__("Recall@k", threshold)

    def _calculate(self, retrieved: list[str], expected: list[str], graded: dict[str, int], k: int) -> float:
        return recall_at_k(retrieved, expected, k)


class PrecisionAtKMetric(RetrievalMetric):
    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__("Precision@k", threshold)

    def _calculate(self, retrieved: list[str], expected: list[str], graded: dict[str, int], k: int) -> float:
        return precision_at_k(retrieved, expected, k)


class HitRateAtKMetric(RetrievalMetric):
    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__("Hit Rate@k", threshold)

    def _calculate(self, retrieved: list[str], expected: list[str], graded: dict[str, int], k: int) -> float:
        return hit_rate_at_k(retrieved, expected, k)


class MRRMetric(RetrievalMetric):
    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__("MRR", threshold)

    def _calculate(self, retrieved: list[str], expected: list[str], graded: dict[str, int], k: int) -> float:
        return reciprocal_rank(retrieved, expected, k)


class NDCGAtKMetric(RetrievalMetric):
    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__("nDCG@k", threshold)

    def _calculate(self, retrieved: list[str], expected: list[str], graded: dict[str, int], k: int) -> float:
        return ndcg_at_k(retrieved, graded, k)


class DuplicateFreeRateAtKMetric(_DeterministicMetric):
    """Assert that every observed top-k result is a unique document."""

    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__("Duplicate-free rate@k", threshold)

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        data = self._metadata(test_case)
        retrieved = data.get("retrieved_document_ids", [])
        k = validate_k(data.get("k", DEFAULT_K))

        self.score = duplicate_free_rate_at_k(retrieved, k)
        self.reason = f"Duplicate-free rate@k={self.score:.6f}"
        self.success = self.is_successful()

        return self.score


class SourceDocumentProvenanceMetric(_DeterministicMetric):
    """Validate source identity, excluding empty negative retrievals as N/A."""

    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__("Source document ID provenance", threshold)

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        data = self._metadata(test_case)
        actual = list(data.get("retrieved_document_ids", []))
        expected = stable_unique_ids(data.get("ground_truth_document_ids", []))

        # Provenance verifies source metadata agrees with the retrieved identity.
        # Ranking and relevance are evaluated separately.
        rows = data.get("retrieved_metadata", [])
        self.applicable = True
        if not actual:
            if not expected:
                self.score = 1.0
                self.applicable = False
                self.success = None
                self.reason = f"not applicable: empty retrieval for negative case; expected={expected!r}"
                return self.score
            self.score = 0.0
        else:
            self.applicable = True
            self.score = float(
                len(rows) == len(actual)
                and all(
                    isinstance(row, Mapping) and row.get("source") == doc_id
                    for doc_id, row in zip(actual, rows)
                )
            )

        self.reason = f"actual={actual!r}; expected={expected!r}"
        self.success = self.is_successful()
        return self.score


class MetadataFidelityMetric(_DeterministicMetric):
    """Verify that expected metadata filters match the corresponding retrieved documents."""

    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__("Metadata key-value fidelity", threshold)

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        data = self._metadata(test_case)
        raw_rows = list(data.get("retrieved_metadata", []))
        actual_rows = [dict(row) for row in raw_rows if isinstance(row, Mapping)]
        if len(actual_rows) != len(raw_rows):
            self.score = 0.0
            self.reason = "Retrieved metadata contains non-mapping rows"
            self.success = self.is_successful()
            return self.score
        expected_rows = data.get("expected_metadata")
        filters = data.get("expected_metadata_filters", {}) or {}

        retrieved_ids = data.get("retrieved_document_ids", [])
        expected_ids = set(data.get("ground_truth_document_ids", []))
        by_id = dict(zip(retrieved_ids, actual_rows))
        if expected_rows is not None:
            # Expected rows are full metadata records; each must match the correct source.
            self.score = float(bool(expected_rows) and len(actual_rows) == len(retrieved_ids) and all(
                isinstance(row, Mapping)
                and row.get("source") in by_id
                and exact_json_equal(by_id[row["source"]], dict(row))
                for row in expected_rows
            ))
        elif filters and expected_ids:
            # Validate metadata on the expected documents actually retrieved.
            self.score = float(all(
                doc_id in by_id and all(
                    key in by_id[doc_id] and exact_json_equal(by_id[doc_id][key], value)
                    for key, value in filters.items()
                ) for doc_id in expected_ids))
        else:
            # No case-specific metadata expectation: provenance still validates source IDs.
            self.score = float(len(actual_rows) == len(retrieved_ids))

        self.reason = f"actual_metadata={actual_rows!r}; expected={expected_rows if expected_rows is not None else filters!r}"
        self.success = self.is_successful()
        return self.score


class SemanticDocumentRelevanceMetric(GEval):
    """GEval judge returning an integer semantic retrieval grade from 0 to 3."""

    def __init__(self, model: str | Any | None = None, threshold: float = 2.0) -> None:
        rubric = [
            Rubric(
                score_range=(0, 0),
                expected_outcome="No retrieved document is semantically relevant to the query or expected evidence.",
            ),
            Rubric(
                score_range=(1, 1),
                expected_outcome="Only weak or incidental relevance is present; key evidence is missing or mostly irrelevant.",
            ),
            Rubric(
                score_range=(2, 2),
                expected_outcome="The retrieval is substantially relevant but has a meaningful coverage or ranking defect.",
            ),
            Rubric(
                score_range=(3, 3),
                expected_outcome="The retrieved documents provide the expected relevant evidence with strong coverage and ranking.",
            ),
        ]
        super().__init__(
            name="Semantic document relevance (0-3)",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            rubric=rubric,
            evaluation_steps=[
                "Read the query, retrieved contexts, and the JSON golden relevance map.",
                "Judge semantic relevance and evidence coverage, not lexical overlap alone.",
                "Assign exactly one integer score: 0, 1, 2, or 3.",
                "Return only valid JSON with integer score and concise reason keys.",
            ],
            model=model,
            threshold=threshold,
            async_mode=False,
            strict_mode=False,
            verbose_mode=False,
        )
        self.threshold = float(threshold)
        self.evaluation_cost = 0.0

    def _check_empty_ground_truth(self, test_case: LLMTestCase) -> bool:
        data = _DeterministicMetric._metadata(test_case)
        ground_truth = data.get("ground_truth_document_ids", [])
        retrieved = data.get("retrieved_document_ids", [])

        if not ground_truth and not retrieved:
            self.score = 3.0
            self.reason = "No retrieval context for an unsupported query."
            self.success = self.is_successful()
            return True
        return False

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        self.evaluation_cost = 0.0

        if self._check_empty_ground_truth(test_case):
            return self.score  # type: ignore[return-value]

        raw_score, reason = self._evaluate(test_case)

        score = int(round(float(raw_score)))
        if score < 0 or score > SEMANTIC_RELEVANCE_MAX:
            raise ValueError(
                f"GEval returned an invalid semantic relevance score: {raw_score!r}"
            )

        self.score = float(score)
        self.reason = str(reason)
        self.success = self.is_successful()

        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        if self._check_empty_ground_truth(test_case):
            return self.score  # type: ignore[return-value]

        if not self.evaluation_steps:
            self.evaluation_steps = await self._a_generate_evaluation_steps()
        raw_score, reason = await self._a_evaluate(test_case)
        score = int(round(float(raw_score)))
        if score < 0 or score > SEMANTIC_RELEVANCE_MAX:
            raise ValueError(f"GEval returned an invalid semantic relevance score: {raw_score!r}")
        self.score = float(score)
        self.reason = str(reason)
        self.success = self.is_successful()
        return self.score


__all__ = [
    "RetrievedDocument",
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "duplicate_free_rate_at_k",
    "RetrievalMetric",
    "RecallAtKMetric",
    "PrecisionAtKMetric",
    "HitRateAtKMetric",
    "MRRMetric",
    "NDCGAtKMetric",
    "DuplicateFreeRateAtKMetric",
    "SourceDocumentProvenanceMetric",
    "MetadataFidelityMetric",
    "SemanticDocumentRelevanceMetric",
]
