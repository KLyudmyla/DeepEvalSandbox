"""Layer 1 checks for idempotent duplicate uploads."""

from evals.adapters.layer1_adapter import get_records, ingest_text


def test_reuploading_the_same_document_does_not_create_a_duplicate(
    empty_layer1_agent,
):
    content = "Overtime limits are capped at up to 3 hours per week."
    source = "duplicate.txt"

    first_result = ingest_text(empty_layer1_agent, content, source)
    second_result = ingest_text(empty_layer1_agent, content, source)
    records = get_records(empty_layer1_agent)

    assert first_result["status"] == "success"
    assert second_result["status"] == "success"
    assert first_result["error"] is None
    assert second_result["error"] is None
    assert len(records) == 1
    assert records[0]["page_content"] == content


# Expected to be failed as no duplication mechanism is implemented
def test_duplicate_detection_uses_content_even_when_source_metadata_differs(
    empty_layer1_agent,
):
    content = "A policy with identical content."

    ingest_text(empty_layer1_agent, content, "source-a.txt")
    ingest_text(empty_layer1_agent, content, "source-b.txt")
    records = get_records(empty_layer1_agent)

    assert len(records) == 1
    assert records[0]["page_content"] == content


# If the document is changed, source should remain the same, but content is updated
def test_changed_content_with_the_same_source_has_overwritten_content(
    empty_layer1_agent,
):
    first_content = "The policy applies to full-time employees."
    second_content = "The policy applies to full-time and part-time employees."
    source = "policy.txt"

    ingest_text(empty_layer1_agent, first_content, source)
    ingest_text(empty_layer1_agent, second_content, source)
    records = get_records(empty_layer1_agent)

    assert len(records) == 1
    assert records[0]["page_content"] == second_content
