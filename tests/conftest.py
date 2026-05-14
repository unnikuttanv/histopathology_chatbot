import matplotlib
matplotlib.use("Agg")  # headless backend for tests

import pytest


@pytest.fixture(autouse=True)
def _disable_dotenv(monkeypatch):
    # config.py calls load_dotenv() at module load, and importlib.reload(config)
    # re-runs it — which would re-inject the developer's real .env values and
    # defeat monkeypatch.delenv(). No-op dotenv so tests stay hermetic.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: False)
