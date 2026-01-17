#!/usr/bin/env python3
"""
Unified test runner with per-test conclusions.
"""
from __future__ import annotations

import sys
import pytest


class ResultPlugin:
    def __init__(self) -> None:
        self.results: list[tuple[str, str]] = []

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call":
            self.results.append((report.nodeid, report.outcome))


def main() -> int:
    plugin = ResultPlugin()
    exit_code = pytest.main(["-q", "tests"], plugins=[plugin])

    print("\nTest conclusions:")
    for nodeid, outcome in plugin.results:
        print(f"- {nodeid}: {outcome.upper()}")

    overall = "PASSED" if exit_code == 0 else "FAILED"
    print(f"\nOverall: {overall}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
