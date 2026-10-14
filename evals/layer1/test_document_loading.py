"""Layer 1 checks for the basic document upload path."""

from evals.adapters.layer1_adapter import get_records, ingest_file


def test_valid_document_is_uploaded_as_one_record(
    empty_layer1_agent,
    layer1_knowledge_base_dir,
):
    fixture = layer1_knowledge_base_dir / "valid_document.txt"

    result = ingest_file(empty_layer1_agent, fixture)
    records = get_records(empty_layer1_agent)

    assert result["status"] == "success"
    assert result["error"] is None
    assert result["before_count"] == 0
    assert result["after_count"] == 1
    assert result["records_added"] == 1
    assert len(records) == 1
    assert records[0]["metadata"]["source"] == fixture.name
