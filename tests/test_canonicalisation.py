from functools import partial

import pytest

from rarg_python_patterns import FrozenKey, freeze, register_freezer
from rarg_python_patterns.multiton.canonicalisation import normalise_args


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


class Defaulted:
  def __init__(self, a, b=10):
    self.a, self.b = a, b


class AllDefaulted:
  def __init__(self, a=10, b=20):
    self.a, self.b = a, b


def test_normalise_args_function():
  """Plain functions: keywords move into positionals, defaults are filled."""

  def fn(a, b=10): ...

  assert normalise_args(fn, (1,), {}) == ((1, 10), {})
  assert normalise_args(fn, (1,), {"b": 10}) == ((1, 10), {})
  assert normalise_args(fn, (), {"a": 1, "b": 2}) == ((1, 2), {})


def test_normalise_args_class_factory():
  """Classes: ``self`` is not counted as a positional slot."""
  assert normalise_args(Defaulted, (1,), {}) == ((1, 10), {})
  assert normalise_args(Defaulted, (1, 10), {}) == ((1, 10), {})
  assert normalise_args(Defaulted, (1,), {"b": 10}) == ((1, 10), {})
  assert normalise_args(AllDefaulted, (1,), {}) == ((1, 20), {})


def test_normalise_args_bound_method():
  """Bound methods/classmethods: ``self``/``cls`` is excluded from binding."""

  class Factory:
    @classmethod
    def create(cls, a, b=3.0):
      return (a, b)

  assert normalise_args(Factory.create, (2.0,), {}) == ((2.0, 3.0), {})
  assert normalise_args(Factory.create, (2.0,), {"b": 3.0}) == ((2.0, 3.0), {})


def test_normalise_args_partial():
  """functools.partial: only the residual signature is bound."""
  assert normalise_args(partial(Defaulted, 1), (), {}) == ((10,), {})
  assert normalise_args(partial(Defaulted, 1), (2,), {}) == ((2,), {})


def test_normalise_args_keyword_only():
  """Keyword-only defaults are filled, canonicalising equivalent calls."""

  def fn(a, *, b=10): ...

  assert normalise_args(fn, (1,), {}) == ((1,), {"b": 10})
  assert normalise_args(fn, (1,), {"b": 10}) == ((1,), {"b": 10})


def test_normalise_args_idempotent():
  """Re-normalising normalised output is a fixed point (pickle round-trip)."""

  def fn(a, *rest, b=10, **extra): ...

  for factory, args, kw in [
    (Defaulted, (1,), {}),
    (fn, (1, 2, 3), {"c": 4}),
  ]:
    nargs, nkw = normalise_args(factory, args, kw)
    assert normalise_args(factory, nargs, nkw) == (nargs, nkw)


def test_normalise_args_invalid_call_raises():
  """Invalid calls fail at normalisation time with a TypeError."""
  with pytest.raises(TypeError):
    normalise_args(Defaulted, (1, 2, 3), {})
  with pytest.raises(TypeError):
    normalise_args(Defaulted, (), {})


def test_normalise_args_non_introspectable_passthrough():
  """Callables without a retrievable signature pass arguments through."""
  args, kw = normalise_args(dict, (), {"a": 1})
  assert (args, kw) == ((), {"a": 1})
