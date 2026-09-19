import pytest

from rarg_python_patterns import Multiton


@pytest.fixture(autouse=True)
def clear_multitons():
  """Ensure a clean cache, heap, holds and key locks before and after each test."""
  Multiton.clear_cache()
  yield
  Multiton.clear_cache()
