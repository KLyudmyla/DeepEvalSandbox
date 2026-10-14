"""Layer 1 checks for UTF-8 file handling."""

import pytest

from evals.adapters.layer1_adapter import get_records, ingest_file


def test_invalid_utf8_file_is_rejected_before_ingestion(
    empty_layer1_agent,
    tmp_path,
):
    invalid_file = tmp_path / "invalid-utf8.txt"
    invalid_file.write_bytes(b"valid prefix\xff\xfe invalid suffix")

    with pytest.raises(UnicodeDecodeError):
        ingest_file(empty_layer1_agent, invalid_file)

    assert get_records(empty_layer1_agent) == []


def test_valid_utf8_file_preserves_non_ascii_text(
    empty_layer1_agent,
    tmp_path,
):
    valid_file = tmp_path / "utf8.txt"
    expected = "Policy: café — 東京\n"
    valid_file.write_text(expected, encoding="utf-8")

    ingest_file(empty_layer1_agent, valid_file)
    records = get_records(empty_layer1_agent)

    assert len(records) == 1
    assert records[0]["page_content"] == expected
    assert records[0]["metadata"]["source"] == valid_file.name
