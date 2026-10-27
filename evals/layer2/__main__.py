"""Command-line entry point for an independent Layer 2 evaluation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from app.agent import DocumentSearchAgent

from evals.datasets.golden_dataset_l2 import GOLDEN_DATASET
from .config import DEFAULT_JUDGE_MODEL
from .run_layer2 import run_layer2_evaluation


def _default_report_name() -> str:
    """Return a unique, filesystem-safe default report filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"layer2_report_{timestamp}.json"


_INDEX_CACHE_VERSION = 1
_INDEX_COLLECTION_NAME = "local_documents"


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def _embedding_configuration(embedding: Any) -> dict[str, Any]:
    """Return non-secret settings that affect generated document vectors."""
    settings: dict[str, Any] = {
        "class": f"{type(embedding).__module__}.{type(embedding).__name__}",
    }
    for name in (
        "model",
        "dimensions",
        "deployment",
        "api_version",
        "base_url",
        "openai_api_base",
        "azure_endpoint",
        "openai_api_type",
        "organization",
        "check_embedding_ctx_length",
        "embedding_ctx_length",
        "tiktoken_enabled",
        "skip_empty",
        "chunk_size",
    ):
        value = getattr(embedding, name, None)
        if value is not None:
            settings[name] = str(value)
    return settings


def _corpus_inventory(knowledge_base: Path) -> list[dict[str, str]]:
    paths = sorted(knowledge_base.glob("*.txt"), key=lambda path: path.name)
    if not paths:
        raise RuntimeError(f"No knowledge-base .txt files found in {knowledge_base}")
    return [
        {
            "source": path.name,
            "content_sha256": hashlib.sha256(
                path.read_text(encoding="utf-8").encode("utf-8")
            ).hexdigest(),
        }
        for path in paths
    ]


def _index_manifest(knowledge_base: Path, embedding: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cache_version": _INDEX_CACHE_VERSION,
        "corpus": _corpus_inventory(knowledge_base),
        "embedding": _embedding_configuration(embedding),
        "index": {
            "collection_name": _INDEX_COLLECTION_NAME,
            "distance": "l2",
            "retrieval": "similarity_search",
            "chromadb": _package_version("chromadb"),
            "langchain_chroma": _package_version("langchain-chroma"),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "fingerprint": hashlib.sha256(encoded).hexdigest()}


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _persistent_index_is_valid(store: Any, manifest: Mapping[str, Any]) -> bool:
    """Validate both the cache manifest and the stored source/content inventory."""
    try:
        rows = store.get(include=["documents", "metadatas"])
        documents = rows.get("documents") or []
        metadatas = rows.get("metadatas") or []
        actual = sorted(
            [
                {
                    "source": str(metadata.get("source", "")),
                    "content_sha256": hashlib.sha256(str(document).encode("utf-8")).hexdigest(),
                }
                for document, metadata in zip(documents, metadatas)
                if isinstance(metadata, Mapping)
            ],
            key=lambda row: (row["source"], row["content_sha256"]),
        )
        expected = sorted(
            manifest.get("corpus", []),
            key=lambda row: (str(row.get("source", "")), str(row.get("content_sha256", "")))
            if isinstance(row, Mapping) else ("", ""),
        )
        return actual == expected
    except Exception:
        return False


def _build_cached_index(
    agent: Any,
    knowledge_base: Path,
    *,
    cache_root: Path,
) -> tuple[Any, dict[str, Any]]:
    """Load a fingerprint-matched Chroma index or build it exactly once.

    A cache is reusable only when the corpus bytes, embedding settings, Chroma
    versions, and retrieval/index settings all match.  A missing or invalid
    manifest is rebuilt; it is never treated as a valid partial index.
    """
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    manifest = _index_manifest(knowledge_base, agent.embeddings)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / str(manifest["fingerprint"])
    manifest_path = cache_path / "manifest.json"
    cached_manifest = _read_manifest(manifest_path)

    store = None
    if cached_manifest == manifest:
        try:
            store = Chroma(
                collection_name=_INDEX_COLLECTION_NAME,
                persist_directory=str(cache_path),
                embedding_function=agent.embeddings,
            )
            if _persistent_index_is_valid(store, manifest):
                return store, {
                    "cache_hit": True,
                    "fingerprint": manifest["fingerprint"],
                    "corpus_documents": len(manifest["corpus"]),
                    "cache_path": str(cache_path),
                }
        except Exception:
            pass
        # The path is for this exact fingerprint, but its contents are not
        # trustworthy. Reset it before rebuilding rather than serving stale data.
        try:
            if store is None:
                raise RuntimeError("Unable to open cached Chroma index")
            store.reset_collection()
            manifest_path.unlink(missing_ok=True)
        except Exception:
            shutil.rmtree(cache_path)
    elif cache_path.exists():
        # A manifest mismatch means this directory cannot be reused safely.
        shutil.rmtree(cache_path)

    cache_path.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=_INDEX_COLLECTION_NAME,
        persist_directory=str(cache_path),
        embedding_function=agent.embeddings,
    )
    documents = []
    for entry in manifest["corpus"]:
        path = knowledge_base / str(entry["source"])
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.name},
            )
        )
    store.add_documents(documents, ids=[entry["source"] for entry in manifest["corpus"]])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return store, {
        "cache_hit": False,
        "fingerprint": manifest["fingerprint"],
        "corpus_documents": len(manifest["corpus"]),
        "cache_path": str(cache_path),
    }


def _build_agent(
    api_key: str,
    *,
    index_cache_dir: str | Path | None = None,
) -> Any:
    from app.agent import DocumentSearchAgent

    agent = DocumentSearchAgent(openai_api_key=api_key)
    knowledge_base = Path(__file__).resolve().parents[2] / "app" / "knowledge_base"
    cache_root = Path(index_cache_dir) if index_cache_dir is not None else (
        Path(__file__).resolve().parents[2] / ".deepeval" / "layer2" / "chroma"
    )
    store, index_info = _build_cached_index(
        agent,
        knowledge_base,
        cache_root=cache_root,
    )
    # The agent's tool and retrieve_documents method read self.vector_store at
    # call time, so replacing the initial empty in-memory store preserves the
    # real application retrieval path while enabling safe persistence.
    agent.vector_store = store
    agent._layer2_index_info = index_info

    return agent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Layer 2 retrieval evaluation harness.")
    parser.add_argument(
        "--report",
        default=None,
        help="Path for the unified JSON report (default: layer2_report_<timestamp>.json).",
    )
    judge_group = parser.add_mutually_exclusive_group()
    judge_group.add_argument(
        "--with-judge",
        dest="with_judge",
        action="store_true",
        default=True,
        help=f"Run the GEval semantic relevance judge (default: {DEFAULT_JUDGE_MODEL}).",
    )
    judge_group.add_argument(
        "--without-judge",
        dest="with_judge",
        action="store_false",
        help="Run deterministic retrieval metrics only; negative-query semantic coverage is not gated.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="DeepEval judge model name when --with-judge is used.",
    )
    parser.add_argument(
        "--max-judge-cost-usd",
        type=float,
        default=None,
        help="Fail safely if known GEval cost exceeds this USD budget; unknown cost is not treated as zero.",
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
        fail_on_threshold=False,
        max_judge_cost_usd=args.max_judge_cost_usd,
    )
    for query_type in ("answerable", "unanswerable"):
        cases = [
            case for case in report["cases"]
            if case["query_type"] == query_type
        ]

        total = len(cases)
        passed = sum(case["success"] is True for case in cases)
        failed = total - passed
        pass_rate = (passed / total * 100) if total else 0.0

        print(
            f"{query_type.capitalize()}: "
            f"{total} cases | "
            f"Passed: {passed} | "
            f"Failed: {failed} | "
            f"Pass rate: {pass_rate:.1f}%"
        )

    print(f"\nOverall: {'PASSED' if report['passed'] else 'FAILED'}")
    print(f"Report: {Path(report_path).resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
