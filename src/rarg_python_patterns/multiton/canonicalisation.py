import inspect
from collections.abc import Callable, Hashable, Mapping, Sequence, Set
from functools import singledispatch
from typing import Any, Dict, Tuple

try:
  from numpy import ndarray
except ImportError:
  # numpy is an optional dependency. Without it, ndarray factory arguments
  # simply fall through to the default freezer.
  ndarray = None  # type: ignore[assignment,misc]


@singledispatch
def _registered_freeze(arg: Any) -> Any:
  """Fallback freezer for types without a registered handler.

  The default leaves the argument unchanged, matching the behaviour of
  values that are already immutable and hashable.
  """
  return arg


def register_freezer(
  cls: type,
) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
  """Register a freezer for ``cls`` (and its subclasses).

  The decorated function is consulted by :func:`freeze` for instances of
  ``cls``. Dispatch is by type via :func:`functools.singledispatch`, so
  lookup is O(1) (MRO-cached) rather than a linear scan over freezers.

  Args:
    cls: the type the decorated freezer handles.

  Returns:
    A decorator that registers and returns the freezer unchanged.
  """
  return _registered_freeze.register(cls)


def freeze(arg: Any) -> Any:
  """Recursively convert argument into an immutable representation"""
  if isinstance(arg, (str, bytes)):
    # str and bytes are sequences, return early to avoid tuplification
    return arg

  if isinstance(arg, slice):
    return (arg.start, arg.stop, arg.step)
  elif isinstance(arg, Sequence):
    return tuple(map(freeze, arg))
  elif isinstance(arg, Set):
    return frozenset(map(freeze, arg))
  elif isinstance(arg, Mapping):
    return frozenset((k, freeze(v)) for k, v in arg.items())
  else:
    # Abstract-container handling above takes precedence; concrete types
    # (e.g. ndarray) are resolved through the freezer registry.
    return _registered_freeze(arg)


if ndarray is not None:

  @register_freezer(ndarray)
  def _freeze_ndarray(arg: Any) -> Tuple[bytes, Tuple[int, ...], str]:
    return (arg.data.tobytes(), arg.shape, arg.dtype.char)


class FrozenKey(Hashable):
  """Converts args and kwargs into an immutable, hashable representation"""

  __slots__ = ("_frozen", "_hashvalue")
  _frozen: Tuple[Any, ...]
  _hashvalue: int

  def __init__(self, *args, **kw):
    self._frozen = freeze(args + (kw,))
    self._hashvalue = hash(self._frozen)

  @property
  def frozen(self) -> Tuple[Any, ...]:
    return self._frozen

  def __hash__(self) -> int:
    return self._hashvalue

  def __eq__(self, other) -> bool:
    if not isinstance(other, FrozenKey):
      return NotImplemented
    return self._hashvalue == other._hashvalue and self._frozen == other._frozen

  def __str__(self) -> str:
    return f"FrozenKey({self._hashvalue})"

  __repr__ = __str__


def normalise_args(
  factory: Callable, args, kw
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
  """Normalise the args and keywords used to call a function.

  Binding through :func:`inspect.signature` canonicalises equivalent calls:
  keywords passing positional parameters move into the positional arguments,
  and omitted defaults (positional and keyword-only) are filled in. Classes,
  bound methods, :func:`functools.partial` objects and callable instances are
  all handled uniformly, with implicit/bound parameters (``self``/``cls``)
  excluded from binding.

  Args:
    factory: factory callable
    args: positional arguments
    kw: keyword arguments

  Returns:
    tuple containing the normalised positional arguments and keyword arguments

  Raises:
    TypeError: if ``args``/``kw`` are not a valid call to ``factory``. This
      surfaces at :class:`Multiton` construction rather than at first
      ``instance`` access.

  Non-introspectable callables (e.g. some builtins) are left un-normalised:
  their arguments pass through unchanged, giving consistent (if not
  canonical) cache keys.
  """
  try:
    sig = inspect.signature(factory)
  except (TypeError, ValueError):
    return tuple(args), kw

  bound = sig.bind(*args, **kw)
  bound.apply_defaults()
  return bound.args, bound.kwargs
