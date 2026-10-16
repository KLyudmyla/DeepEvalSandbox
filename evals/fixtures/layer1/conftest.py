"""Shared fixtures for Layer 1 ingestion evaluations.

The fixtures create a new application agent for every test. This is
important because the application currently uses an in-memory Chroma store;
sharing an agent would make test order affect the results.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from evals.adapters.layer1_adapter import create_agent, get_records


PROJECT_ROOT = Path(__file__).resolve().parents[3]
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "app" / "knowledge_base"


@pytest.fixture(scope="session")
def layer1_knowledge_base_dir() -> Path:
    """Return the application knowledge-base directory used by Layer 1."""
    return KNOWLEDGE_BASE_DIR


@pytest.fixture(scope="session")
def layer1_api_key() -> str:
    """Return the API key required by the real application agent."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is required for Layer 1 ingestion tests")
    return api_key


@pytest.fixture
def empty_layer1_agent(layer1_api_key):
    """Create a fresh agent and assert that its store starts empty."""
    agent = create_agent(layer1_api_key)
    assert get_records(agent) == []
    return agent
