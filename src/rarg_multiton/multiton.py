from __future__ import annotations

import heapq
import itertools
import math
import numbers
import time
from threading import RLock
from typing import Any, Callable, ClassVar, Dict, Generic, List, Tuple, TypeVar

from rarg_multiton.canonicalisation import FrozenKey, normalise_args

T = TypeVar("T")

# (object, last_accessed_monotonic, ttl_seconds, seq)
_CacheEntry = Tuple[Any, float, float, int]
# (expiry_monotonic, seq, key) — min-heap ordered by expiry
_HeapEntry = Tuple[float, int, FrozenKey]


class Multiton(Generic[T]):
  """Implementation of the Multiton pattern with TTL-based cache expiry.

  See https://en.wikipedia.org/wiki/Multiton_pattern for an overview.

  Multiton's are hashable, equality-comparable and pickleable as long
  as the supplied arguments also support these properties.

  Cached instances expire after ``ttl`` seconds of inactivity. Accessing
  ``instance`` resets the TTL for that entry. All expired entries are
  swept from the cache on every ``instance`` access via a min-heap ordered
  by expiry time, so only genuinely expired entries are visited.

  A ``ttl`` of ``math.inf`` (set via ``with_ttl(math.inf)`` or the
  ``with_infinite_ttl()`` shorthand) makes an entry eternal: it never
  expires and is only removed by ``release()``. Eternal entries are still
  pushed onto the heap but their ``inf`` deadline never satisfies the sweep
  condition, so the heap self-compacts (discarding ``inf`` and stale tuples)
  whenever it grows much larger than the live cache.

  .. code-block:: python

    # Factory function creating a resource
    def open_connection(url: str, timeout: float = 1.0) -> Connection:
      ...

    # Create a multiton representing a resource (TTL defaults to _DEFAULT_TTL)
    resource = Multiton(open_connection, "https://www.python.org", timeout=10.0)

    # The resource is only created when the instance attribute is accessed
    response = resource.instance.request("GET", "/foo/bar.html")
  """

  # Class variables
  _DEFAULT_TTL: ClassVar[float] = 300.0
  _INSTANCE_CACHE: ClassVar[Dict[FrozenKey, _CacheEntry]] = {}
  _EXPIRY_HEAP: ClassVar[List[_HeapEntry]] = []
  _SEQUENCE: ClassVar[itertools.count] = itertools.count()
  _INSTANCE_LOCK: ClassVar[RLock] = RLock()
  # Compact the heap once it exceeds max(_HEAP_COMPACT_MIN,
  # _HEAP_COMPACT_FACTOR * len(cache)) entries, bounding stale/eternal bloat.
  _HEAP_COMPACT_MIN: ClassVar[float] = 32
  _HEAP_COMPACT_FACTOR: ClassVar[float] = 2.0

  __slots__ = ("_factory", "_args", "_kw", "_key", "_ttl")

  # Instance variables
  _factory: Callable[..., T]
  _args: Tuple[Any, ...]
  _kw: Dict[str, Any]
  _key: FrozenKey
  _ttl: float

  def __init__(self, factory: Callable[..., T], *args, **kw):
    """Create a Multiton with the factory function and arguments
    necessary for creating the underlying object instance.

    Arguments:
      factory: A factory function
      args: Arguments passed to the factory function
      kw: Keyword arguments passed to the factory function
    """
    self._factory = factory
    self._args, self._kw = normalise_args(factory, args, kw)
    self._key = FrozenKey(factory, *self._args, **self._kw)
    self._ttl = self._DEFAULT_TTL

  def with_args(self, *, ttl: float) -> Multiton[T]:
    """Set per-instance cache options and return ``self`` for chaining.

    Arguments:
      ttl: Time-to-live in seconds for the cached instance. Accessing
        ``instance`` resets the TTL. Only takes effect when this Multiton
        first creates the cache entry; if an entry already exists its TTL
        is not changed. ``math.inf`` makes the entry eternal. Must be a real
        number; ``nan`` is rejected, as it would never satisfy the expiry
        comparison and would leak in the heap.
    """
    if not isinstance(ttl, numbers.Real) or math.isnan(ttl):
      raise ValueError(f"ttl must be a real number, not nan (got {ttl!r})")
    self._ttl = ttl
    return self

  def with_ttl(self, ttl: float) -> Multiton[T]:
    """Set the per-instance TTL and return ``self`` for chaining.

    Arguments:
      ttl: Time-to-live in seconds for the cached instance. ``math.inf``
        makes the entry eternal (never expires; only removed by ``release()``).
        See :meth:`with_args` for the TTL-reset and first-write semantics.
    """
    return self.with_args(ttl=ttl)

  def with_infinite_ttl(self) -> Multiton[T]:
    """Cache the instance forever (until explicitly released).

    Shorthand for ``with_ttl(math.inf)``; returns ``self`` for chaining.
    """
    return self.with_args(ttl=math.inf)

  @staticmethod
  def from_reduce_args(factory: Callable[..., T], args, kw, ttl: float) -> Multiton[T]:
    """Helper method for reconstructing a Multiton from arg and kw objects"""
    return Multiton[T](factory, *args, **kw).with_args(ttl=ttl)

  def __reduce__(self) -> Tuple[Callable, Tuple[Any, ...]]:
    return (Multiton.from_reduce_args, (self._factory, self._args, self._kw, self._ttl))

  def __hash__(self) -> int:
    return hash(self._key)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, Multiton):
      return NotImplemented
    return self._key == other._key

  @classmethod
  def _write_entry(cls, key: FrozenKey, obj: Any, ttl: float) -> None:
    """Write a cache entry and push the corresponding heap entry.
    Must be called under the lock."""
    seq = next(cls._SEQUENCE)
    now = time.monotonic()
    cls._INSTANCE_CACHE[key] = (obj, now, ttl, seq)
    heapq.heappush(cls._EXPIRY_HEAP, (now + ttl, seq, key))

  @classmethod
  def _compact_heap(cls) -> None:
    """Rebuild the heap from live, finite-TTL cache entries.
    Must be called under the lock.

    Iterating the cache and re-emitting one tuple per live key (with that
    entry's current seq) structurally discards both eternal (``inf``) tuples
    and any finite tuple whose seq no longer matches its cache entry — i.e.
    stale entries left by TTL resets or releases that the sweep loop can't
    reach. ``last + ttl`` reproduces the exact expiry pushed for that seq.
    """
    cls._EXPIRY_HEAP[:] = [
      (last + ttl, seq, key)
      for key, (_, last, ttl, seq) in cls._INSTANCE_CACHE.items()
      if not math.isinf(ttl)
    ]
    heapq.heapify(cls._EXPIRY_HEAP)

  @classmethod
  def _purge_expired(cls) -> None:
    """Remove expired entries from the cache using the heap.
    Must be called under the lock.

    Pops heap entries whose deadline has passed, discarding those whose seq
    no longer matches the cache (stale due to TTL reset or release). Eternal
    (``inf``) tuples never satisfy the deadline, so once the heap grows much
    larger than the live cache it is compacted to reclaim them along with any
    stale finite tuples.
    """
    now = time.monotonic()
    while cls._EXPIRY_HEAP and cls._EXPIRY_HEAP[0][0] <= now:
      _, seq, key = heapq.heappop(cls._EXPIRY_HEAP)
      entry = cls._INSTANCE_CACHE.get(key)
      if entry is None or entry[3] != seq:
        # stale: key was released, or TTL was reset since this heap entry was pushed
        continue
      del cls._INSTANCE_CACHE[key]

    threshold = max(
      cls._HEAP_COMPACT_MIN, cls._HEAP_COMPACT_FACTOR * len(cls._INSTANCE_CACHE)
    )
    if len(cls._EXPIRY_HEAP) > threshold:
      cls._compact_heap()

  @property
  def instance(self) -> T:
    """Returns the instance defined by this Multiton, creating it if necessary.

    Expired cache entries are swept on every call via the heap. Accessing a
    live entry resets its TTL.
    """
    with self._INSTANCE_LOCK:
      self._purge_expired()
      entry = self._INSTANCE_CACHE.get(self._key)

      # Reset the TTL
      if entry is not None:
        obj, _, ttl, _ = entry
        self._write_entry(self._key, obj, ttl)
        return obj

      obj = self._factory(*self._args, **self._kw)
      self._write_entry(self._key, obj, self._ttl)
      return obj

  def release(self) -> None:
    """Immediately evict this Multiton's instance from the cache.

    Any Multiton sharing the same key will recreate the instance on next
    access. The corresponding heap entry is left in place and discarded
    as stale during the next purge sweep.
    """
    with self._INSTANCE_LOCK:
      self._INSTANCE_CACHE.pop(self._key, None)

  def __str__(self) -> str:
    return f"Multiton({self._factory})"

  __repr__ = __str__
