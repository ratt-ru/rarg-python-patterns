import pytest

from rarg_python_patterns import Multiton


@pytest.fixture(autouse=True)
def clear_multitons():
  """Ensure a clean cache, heap and key locks before and after each test."""
  Multiton._INSTANCE_CACHE.clear()
  Multiton._EXPIRY_HEAP.clear()
  Multiton._KEY_LOCKS.clear()
  yield
  Multiton._INSTANCE_CACHE.clear()
  Multiton._EXPIRY_HEAP.clear()
  Multiton._KEY_LOCKS.clear()
