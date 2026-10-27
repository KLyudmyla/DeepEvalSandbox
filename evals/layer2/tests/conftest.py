"""Pytest policy for Layer 2.

Live Layer 2 tests are deliberately opt-in.  Importing this module and running
the offline suite must never construct the real agent or call a paid API.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("layer2")
    group.addoption(
        "--run-layer2-live",
        action="store_true",
        default=False,
        help="Collect Layer 2 tests that use the real agent and external APIs.",
    )
    group.addoption(
        "--layer2-with-judge",
        action="store_true",
        default=False,
        help="Deprecated compatibility flag; the GEval judge is enabled by default.",
    )
    group.addoption(
        "--layer2-without-judge",
        action="store_true",
        default=False,
        help="Disable the GEval judge for an explicitly deterministic-only live run.",
    )
    group.addoption(
        "--layer2-judge-model",
        default="gpt-4o-mini",
        help="Judge model used with --layer2-with-judge.",
    )
    group.addoption(
        "--layer2-max-judge-cost-usd",
        type=float,
        default=None,
        help="Fail the live Layer 2 run if known GEval cost exceeds this budget.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: uses the real DocumentSearchAgent and may call paid external APIs",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect live tests unless the caller explicitly opts in.

    Deselecting instead of silently skipping keeps the default run free of
    external calls while making the opt-in boundary visible in pytest output.
    """

    if config.getoption("--run-layer2-live"):
        return

    live_items = [item for item in items if "live" in item.keywords]
    if not live_items:
        return

    config.hook.pytest_deselected(items=live_items)
    items[:] = [item for item in items if item not in live_items]
