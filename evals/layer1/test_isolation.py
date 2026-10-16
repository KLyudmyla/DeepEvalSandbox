"""Layer 1 checks for isolation between independent evaluation runs."""

from evals.adapters.layer1_adapter import create_agent, get_records, ingest_text


def test_a_new_agent_does_not_see_documents_from_another_run(layer1_api_key):
    """Verify that independent agent instances do not share knowledge-base data.

    Requirement: documents uploaded during one evaluation run must not be
    visible in another run. This protects tenant/session boundaries and keeps
    one evaluation from contaminating another evaluation's result.

    Arrange: create two independent agents.
    Act: upload a different source to each agent.
    Assert: each store contains exactly its own source and no foreign source.
    """
    first_agent = create_agent(layer1_api_key)
    second_agent = create_agent(layer1_api_key)

    ingest_text(first_agent, "First run content.", "first-run.txt")
    ingest_text(second_agent, "Second run content.", "second-run.txt")

    first_records = get_records(first_agent)
    second_records = get_records(second_agent)

    assert len(first_records) == 1
    assert len(second_records) == 1
    assert {record["metadata"].get("source") for record in first_records} == {
        "first-run.txt"
    }
    assert {record["metadata"].get("source") for record in second_records} == {
        "second-run.txt"
    }
    assert all(
        record["metadata"].get("source") != "first-run.txt"
        for record in second_records
    )


def test_isolation_allows_multiple_documents_within_one_run(empty_layer1_agent):
    """Verify that isolation does not limit one run to a single document.

    Requirement: an isolated knowledge base must still support multiple
    documents in the same run, preserving each source-to-content association.

    Arrange: start with one empty agent.
    Act: upload two distinct documents to that agent.
    Assert: both documents remain present and neither is overwritten or mixed
    with the other document's metadata.
    """
    first_content = "Alpha"
    second_content = "Beta"
    ingest_text(empty_layer1_agent, first_content, "alpha.txt")
    ingest_text(empty_layer1_agent, second_content, "beta.txt")

    records = get_records(empty_layer1_agent)

    assert len(records) == 2
    content_by_source = {
        record["metadata"].get("source"): record["page_content"]
        for record in records
    }
    assert content_by_source == {
        "alpha.txt": first_content,
        "beta.txt": second_content
    }
