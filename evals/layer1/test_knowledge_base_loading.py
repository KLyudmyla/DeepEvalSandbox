"""Functional Layer 1 checks for application knowledge-base loading."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from evals.adapters.layer1_adapter import get_records


@pytest.fixture
def application_main(layer1_api_key, monkeypatch):
    """Import the application module after the Layer 1 API key is available."""
    monkeypatch.setenv("OPENAI_API_KEY", layer1_api_key)
    return importlib.import_module("app.main")


def test_startup_loads_every_knowledge_base_text_file_with_exact_integrity(
    empty_layer1_agent,
    layer1_knowledge_base_dir,
    application_main,
    monkeypatch,
):
    expected_by_source = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(layer1_knowledge_base_dir.glob("*.txt"))
    }
    assert expected_by_source

    monkeypatch.setattr(application_main, "agent", empty_layer1_agent)
    monkeypatch.setattr(
        application_main,
        "KNOWLEDGE_BASE_DIR",
        layer1_knowledge_base_dir,
    )

    application_main.preload_documents()
    records = get_records(empty_layer1_agent)

    actual_by_source = {
        record["metadata"].get("source"): record["page_content"]
        for record in records
    }
    assert len(records) == len(expected_by_source)
    assert actual_by_source == expected_by_source


def test_startup_loads_only_txt_files(
    empty_layer1_agent,
    tmp_path: Path,
    application_main,
    monkeypatch,
):
    text_file = tmp_path / "accepted.txt"
    text_file.write_text("A supported knowledge-base document.", encoding="utf-8")
    (tmp_path / "ignored.md").write_text(
        "This file must not be ingested.",
        encoding="utf-8",
    )

    monkeypatch.setattr(application_main, "agent", empty_layer1_agent)
    monkeypatch.setattr(application_main, "KNOWLEDGE_BASE_DIR", tmp_path)

    application_main.preload_documents()
    records = get_records(empty_layer1_agent)

    assert len(records) == 1
    assert records[0]["metadata"]["source"] == text_file.name
    assert records[0]["page_content"] == text_file.read_text(encoding="utf-8")


def test_startup_with_an_empty_knowledge_base_leaves_the_store_empty(
    empty_layer1_agent,
    tmp_path: Path,
    application_main,
    monkeypatch,
):
    monkeypatch.setattr(application_main, "agent", empty_layer1_agent)
    monkeypatch.setattr(application_main, "KNOWLEDGE_BASE_DIR", tmp_path)

    application_main.preload_documents()

    assert get_records(empty_layer1_agent) == []


def test_invalid_utf8_knowledge_base_file_is_rejected_before_ingestion(
    empty_layer1_agent,
    tmp_path: Path,
    application_main,
    monkeypatch,
):
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_bytes(b"valid prefix\xff\xfe invalid suffix")

    monkeypatch.setattr(application_main, "agent", empty_layer1_agent)
    monkeypatch.setattr(application_main, "KNOWLEDGE_BASE_DIR", tmp_path)

    with pytest.raises(UnicodeDecodeError):
        application_main.preload_documents()

    assert get_records(empty_layer1_agent) == []


def test_preloaded_knowledge_base_document_is_retrievable(
    empty_layer1_agent,
    layer1_knowledge_base_dir,
    application_main,
    monkeypatch,
):
    monkeypatch.setattr(application_main, "agent", empty_layer1_agent)
    monkeypatch.setattr(
        application_main,
        "KNOWLEDGE_BASE_DIR",
        layer1_knowledge_base_dir,
    )

    application_main.preload_documents()
    retrieved = empty_layer1_agent.retrieve_documents(
        "overtime limits for contract-based employees",
        k=2,
    )

    expected_path = layer1_knowledge_base_dir / "valid_document.txt"
    expected_content = expected_path.read_text(encoding="utf-8")
    assert retrieved
    assert any(
        document.metadata.get("source") == expected_path.name
        and document.page_content == expected_content
        for document in retrieved
    )
