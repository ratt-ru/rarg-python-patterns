import pytest

from rarg_python_patterns import FrozenKey, freeze, register_freezer


def test_freeze_basic_types():
  """Common containers freeze into hashable, comparable representations."""
  assert freeze("abc") == "abc"
  assert freeze(b"abc") == b"abc"
  assert freeze([1, 2, 3]) == (1, 2, 3)
  assert freeze({1, 2}) == frozenset({1, 2})
  assert freeze(slice(1, 10, 2)) == (1, 10, 2)
  assert freeze({"a": [1, 2]}) == frozenset({("a", (1, 2))})

  # Equal inputs hash equally; nested containers are handled
  assert hash(FrozenKey([1, 2], k={"x": 3})) == hash(FrozenKey([1, 2], k={"x": 3}))


def test_freeze_numpy_when_available():
  """When numpy is installed, ndarrays freeze by their bytes/shape/dtype."""
  np = pytest.importorskip("numpy")
  a = np.array([1, 2, 3])
  b = np.array([1, 2, 3])
  c = np.array([1, 2, 4])
  assert FrozenKey(a) == FrozenKey(b)
  assert FrozenKey(a) != FrozenKey(c)


def test_register_freezer_for_custom_type():
  """A registered freezer is consulted by ``freeze`` for its type."""

  class Point:
    def __init__(self, x, y):
      self.x = x
      self.y = y

  @register_freezer(Point)
  def _freeze_point(p):
    return (p.x, p.y)

  assert freeze(Point(1, 2)) == (1, 2)
  # Equal points produce equal, hashable keys; distinct points do not.
  assert FrozenKey(Point(1, 2)) == FrozenKey(Point(1, 2))
  assert FrozenKey(Point(1, 2)) != FrozenKey(Point(3, 4))
