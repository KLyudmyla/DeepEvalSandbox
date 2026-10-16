"""Layer 1 checks for exact content preservation."""

from evals.adapters.layer1_adapter import get_records, ingest_text


def test_content_is_preserved_byte_for_byte_as_decoded_text(empty_layer1_agent):
    content = "First line\n\nUnicode: café — 東京\nTrailing line"
    source = "integrity.txt"

    result = ingest_text(empty_layer1_agent, content, source)
    records = get_records(empty_layer1_agent)

    assert result["status"] == "success"
    assert result["error"] is None
    matching = [
        record
        for record in records
        if record["metadata"].get("source") == source
    ]
    assert len(matching) == 1
    assert matching[0]["page_content"] == content


def test_content_integrity_is_scoped_to_the_document_source(empty_layer1_agent):
    first_content = "The first source has a unique policy."
    second_content = "The second source has a different policy."

    ingest_text(empty_layer1_agent, first_content, "first.txt")
    ingest_text(empty_layer1_agent, second_content, "second.txt")
    records = get_records(empty_layer1_agent)

    assert len(records) == 2
    content_by_source = {
        record["metadata"].get("source"): record["page_content"]
        for record in records
    }
    assert content_by_source == {
        "first.txt": first_content,
        "second.txt": second_content,
    }
