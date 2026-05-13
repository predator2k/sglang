"""Pytest configuration for Tenstorrent backend tests.

Registers custom markers so that -m simple_backend / -m paged_backend /
-m hardware select only the appropriate subset.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "simple_backend: tests targeting SGLANG_TT_EXECUTION_BACKEND=tt_transformers_single",
    )
    config.addinivalue_line(
        "markers",
        "paged_backend: tests targeting SGLANG_TT_EXECUTION_BACKEND=tt_transformers_paged",
    )
    config.addinivalue_line(
        "markers",
        "hardware: requires a live ttnn mesh",
    )
