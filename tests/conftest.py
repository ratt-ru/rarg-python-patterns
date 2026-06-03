import pytest

from rarg_multiton import Multiton


@pytest.fixture(autouse=True)
def clear_multitons():
  """Ensure a clean cache and heap before and after each test."""
  Multiton._INSTANCE_CACHE.clear()
  Multiton._EXPIRY_HEAP.clear()
  yield
  Multiton._INSTANCE_CACHE.clear()
  Multiton._EXPIRY_HEAP.clear()
