"""Layer 1 checks for source metadata preservation."""

import pytest

from evals.adapters.layer1_adapter import get_records, ingest_text


def test_each_record_preserves_its_exact_source_metadata(empty_layer1_agent):
    first_source = "engineering/stack v2 (EU).txt"
    second_source = "hr policy — remote.txt"
    ingest_text(empty_layer1_agent, "One", first_source)
    ingest_text(empty_layer1_agent, "Two", second_source)
    records = get_records(empty_layer1_agent)

    assert {
        record["metadata"].get("source")
        for record in records
    } == {
        first_source,
        second_source,
    }
    assert len(records) == 2
    assert all(record["metadata"].get("source") for record in records)


@pytest.mark.parametrize("source", ["", " \t"])
def test_blank_source_is_rejected_without_creating_a_record(
    empty_layer1_agent,
    source,
):
    result = ingest_text(
        empty_layer1_agent,
        "A document with no usable provenance.",
        source,
    )

    assert result["status"] == "rejected"
    assert isinstance(result["error"], ValueError)
    assert result["before_count"] == 0
    assert result["after_count"] == 0
    assert get_records(empty_layer1_agent) == []
