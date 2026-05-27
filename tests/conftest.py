from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires real infrastructure credentials in .env)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if not config.getoption("--integration"):
        skip = pytest.mark.skip(reason="pass --integration to run integration tests")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)
