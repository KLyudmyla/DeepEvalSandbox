"""Execute real ingestion and inspect ChromaDB.

This module deliberately keeps the application behavior observable.  It does
not add validation, deduplication, persistence, or cleanup policies that are
not implemented by the application itself.
Isolation is observed by creating a fresh agent for each scenario; this
adapter does not inject a separate storage implementation.
"""

from pathlib import Path
from typing import Any

from app.agent import DocumentSearchAgent


def create_agent(api_key: str) -> DocumentSearchAgent:
    """Create a fresh application agent for one isolated evaluation run."""
    return DocumentSearchAgent(openai_api_key=api_key)


def ingest_text(
    agent: DocumentSearchAgent,
    text: str,
    source: str,
) -> dict[str, Any]:
    """Call the real application ingestion path and capture the result."""
    before = count_records(agent)

    error: Exception | None = None
    try:
        agent.upload_document(text=text, source=source)
    except Exception as exc:  # The evaluator needs to inspect rejected inputs.
        error = exc

    after = count_records(agent)

    return {
        "status": "success" if error is None else "rejected",
        "error": error,
        "before_count": before,
        "after_count": after,
        "records_added": after - before,
    }


def ingest_file(
    agent: DocumentSearchAgent,
    path: Path,
    source: str | None = None,
) -> dict[str, Any]:
    """Read a fixture as UTF-8 and pass it through real ingestion."""
    text = path.read_text(encoding="utf-8")
    return ingest_text(
        agent,
        text=text,
        source=source or path.name,
    )


def get_records(agent: DocumentSearchAgent) -> list[dict[str, Any]]:
    """Return stored Chroma records in an evaluation-friendly form."""
    result = agent.vector_store.get(
        include=["documents", "metadatas"],
    )

    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    ids = result.get("ids", [])

    return [
        {
            "id": record_id,
            "page_content": content,
            "metadata": metadata or {},
        }
        for record_id, content, metadata in zip(ids, documents, metadatas)
    ]


def count_records(agent: DocumentSearchAgent) -> int:
    """Return the number of records currently stored in Chroma."""
    return len(get_records(agent))
