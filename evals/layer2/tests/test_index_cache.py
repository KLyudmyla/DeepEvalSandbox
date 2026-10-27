"""Offline checks for safe Layer 2 index-cache invalidation."""

import sys
from types import ModuleType, SimpleNamespace

from evals.layer2.__main__ import _index_manifest
from evals.layer2.__main__ import _build_cached_index


class _Embedding:
    model = "text-embedding-test"
    dimensions = 1536


def test_index_fingerprint_changes_when_corpus_content_changes(tmp_path):
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")
    first = _index_manifest(tmp_path, _Embedding())

    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    second = _index_manifest(tmp_path, _Embedding())

    assert first["fingerprint"] != second["fingerprint"]


def test_index_fingerprint_changes_when_embedding_configuration_changes(tmp_path):
    (tmp_path / "a.txt").write_text("same", encoding="utf-8")

    first = _index_manifest(tmp_path, _Embedding())
    changed_embedding = _Embedding()
    changed_embedding.model = "text-embedding-other"
    second = _index_manifest(tmp_path, changed_embedding)

    assert first["fingerprint"] != second["fingerprint"]


def test_matching_index_cache_avoids_readding_documents(tmp_path, monkeypatch):
    class _Document:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    class _Chroma:
        stores = {}
        add_calls = 0

        def __init__(self, collection_name, persist_directory, embedding_function):
            self.key = str(persist_directory)
            self.rows = self.stores.setdefault(self.key, [])

        def get(self, include=None):
            return {
                "documents": [row[0] for row in self.rows],
                "metadatas": [row[1] for row in self.rows],
            }

        def add_documents(self, documents, ids):
            type(self).add_calls += 1
            self.rows[:] = [(doc.page_content, doc.metadata) for doc in documents]

        def reset_collection(self):
            self.rows.clear()

    chroma_module = ModuleType("langchain_chroma")
    chroma_module.Chroma = _Chroma
    documents_module = ModuleType("langchain_core.documents")
    documents_module.Document = _Document
    monkeypatch.setitem(sys.modules, "langchain_chroma", chroma_module)
    monkeypatch.setitem(sys.modules, "langchain_core.documents", documents_module)

    knowledge_base = tmp_path / "kb"
    knowledge_base.mkdir()
    (knowledge_base / "a.txt").write_text("content", encoding="utf-8")
    agent = SimpleNamespace(embeddings=_Embedding())

    first_store, first_info = _build_cached_index(
        agent, knowledge_base, cache_root=tmp_path / "cache"
    )
    second_store, second_info = _build_cached_index(
        agent, knowledge_base, cache_root=tmp_path / "cache"
    )

    assert first_info["cache_hit"] is False
    assert second_info["cache_hit"] is True
    assert first_store is not second_store
    assert _Chroma.add_calls == 1
