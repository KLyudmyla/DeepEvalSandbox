"""Layer 1 checks for empty document rejection and store safety."""

import pytest

from evals.adapters.layer1_adapter import get_records, ingest_text


@pytest.mark.parametrize(
    ("content", "source"),
    [
        ("", "empty.txt"),
        (" \n\t ", "whitespace.txt"),
    ],
)

# Expected to fail as there is no validation for the empty document
def test_blank_documents_are_rejected_without_creating_a_record(
    empty_layer1_agent,
    content,
    source,
):
    result = ingest_text(empty_layer1_agent, content, source)

    assert result["status"] == "rejected"
    assert isinstance(result["error"], ValueError)
    assert result["before_count"] == 0
    assert result["after_count"] == 0
    assert result["records_added"] == 0
    assert get_records(empty_layer1_agent) == []


@pytest.mark.parametrize("content", ["", " \n\t "])
def test_rejected_blank_documents_do_not_mutate_existing_records(
    empty_layer1_agent,
    content,
):
    stored_content = "This document must survive a rejected upload."
    ingest_text(empty_layer1_agent, stored_content, "existing.txt")

    result = ingest_text(empty_layer1_agent, content, "blank.txt")

    assert result["status"] == "rejected"
    assert isinstance(result["error"], ValueError)
    records = get_records(empty_layer1_agent)
    assert result["before_count"] == 1
    assert result["after_count"] == 1
    assert result["records_added"] == 0
    matching = [
        record
        for record in records
        if record["metadata"].get("source") == "existing.txt"
    ]
    assert len(matching) == 1
    assert matching[0]["page_content"] == stored_content
