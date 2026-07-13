from __future__ import annotations

import heapq
import itertools
import math
import numbers
import time
import weakref
from threading import RLock
from typing import Any, Callable, ClassVar, Dict, Generic, List, Tuple, TypeVar

from rarg_python_patterns.multiton.canonicalisation import FrozenKey, normalise_args

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

  By default only the factory and its arguments are pickled and the
  receiving process recreates the instance lazily. After
  ``with_serialise_instance()`` the underlying instance is pickled too:
  ``__reduce__`` constructs it if it is not already cached, and unpickling
  seeds the cache with it (unless the key is already cached), so the
  factory never runs in the receiving process — useful when e.g. a
  configuration file should be read once and its contents shipped to
  processes where the file may not exist. The instance itself must then
  be pickleable, and the seeded entry still expires after ``ttl`` of
  inactivity, so consider combining with ``with_infinite_ttl()``.

  Cached instances expire after ``ttl`` seconds of inactivity. Accessing
  ``instance`` resets the TTL for that entry. All expired entries are
  swept from the cache on every ``instance`` access via a min-heap ordered
  by expiry time, so only genuinely expired entries are visited.

  A ``ttl`` of ``math.inf`` (set via ``with_ttl(math.inf)`` or the
  ``with_infinite_ttl()`` shorthand) makes an entry eternal: it never
  expires and is only removed by ``release()``. Eternal entries are never
  pushed onto the heap (an ``inf`` deadline could never satisfy the sweep
  condition anyway), so they cost nothing in heap space. The heap holds
  only finite-TTL tuples and self-compacts to discard stale ones whenever
  it grows much larger than the live cache.

  **Thread safety.** Cache hits acquire only a brief global lock and never
  block behind factory execution. Constructions are serialised per key:
  two threads racing on the same key run the factory once, while threads
  touching other keys proceed unhindered. A factory may freely access
  *other* multitons' ``instance`` during construction, provided the
  dependency graph between factories is acyclic — two factories that
  construct each other from different threads will deadlock.

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
  # Per-key construction locks. Values are weakly referenced: a lock lives
  # exactly as long as some thread is constructing or waiting on its key
  # (those threads hold strong references), then vanishes — released and
  # expired entries cannot leak locks.
  _KEY_LOCKS: ClassVar["weakref.WeakValueDictionary[FrozenKey, Any]"] = (
    weakref.WeakValueDictionary()
  )
  # Compact the heap once it exceeds max(_HEAP_COMPACT_MIN,
  # _HEAP_COMPACT_FACTOR * len(cache)) entries, bounding stale/eternal bloat.
  _HEAP_COMPACT_MIN: ClassVar[float] = 32
  _HEAP_COMPACT_FACTOR: ClassVar[float] = 2.0

  __slots__ = ("_factory", "_args", "_kw", "_key", "_ttl", "_serialise_instance")

  # Instance variables
  _factory: Callable[..., T]
  _args: Tuple[Any, ...]
  _kw: Dict[str, Any]
  _key: FrozenKey
  _ttl: float
  _serialise_instance: bool

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
    self._serialise_instance = False

  def with_args(
    self, *, ttl: float | None = None, serialise_instance: bool | None = None
  ) -> Multiton[T]:
    """Set per-instance cache options and return ``self`` for chaining.

    Options left at ``None`` are unchanged.

    Arguments:
      ttl: Time-to-live in seconds for the cached instance. Accessing
        ``instance`` resets the TTL. Only takes effect when this Multiton
        first creates the cache entry; if an entry already exists its TTL
        is not changed. ``math.inf`` makes the entry eternal. Must be a real
        number; ``nan`` is rejected, as it would never satisfy the expiry
        comparison and would leak in the heap.
      serialise_instance: When true, pickling this Multiton also pickles
        the underlying instance — constructing it first if necessary — and
        unpickling seeds the cache with it (only if the key is not already
        cached), so the factory never runs in the receiving process. The
        instance itself must then be pickleable. The seeded entry still
        expires after ``ttl`` of inactivity; combine with
        ``with_infinite_ttl()`` if the factory cannot run remotely.
    """
    if ttl is not None:
      if not isinstance(ttl, numbers.Real) or math.isnan(ttl):
        raise ValueError(f"ttl must be a real number, not nan (got {ttl!r})")
      self._ttl = ttl
    if serialise_instance is not None:
      self._serialise_instance = bool(serialise_instance)
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

  def with_serialise_instance(self) -> Multiton[T]:
    """Pickle the underlying instance along with this Multiton.

    Shorthand for ``with_args(serialise_instance=True)``; returns ``self``
    for chaining. See :meth:`with_args` for the full semantics.
    """
    return self.with_args(serialise_instance=True)

  @staticmethod
  def from_reduce_args(
    factory: Callable[..., T],
    args,
    kw,
    ttl: float,
    serialise_instance: bool = False,
    instance: Any = None,
  ) -> Multiton[T]:
    """Helper method for reconstructing a Multiton from arg and kw objects.

    When ``serialise_instance`` is true, ``instance`` is the materialised
    object shipped in the pickle; it seeds the cache unless the key is
    already cached (mirroring the first-write TTL semantics)."""
    multiton = Multiton[T](factory, *args, **kw).with_args(
      ttl=ttl, serialise_instance=serialise_instance
    )
    if serialise_instance:
      with Multiton._INSTANCE_LOCK:
        Multiton._purge_expired()
        if multiton._key not in Multiton._INSTANCE_CACHE:
          Multiton._write_entry(multiton._key, instance, ttl)
    return multiton

  def __reduce__(self) -> Tuple[Callable, Tuple[Any, ...]]:
    if self._serialise_instance:
      # Forces construction if the instance is not currently cached, so the
      # pickle always carries a materialised instance.
      return (
        Multiton.from_reduce_args,
        (self._factory, self._args, self._kw, self._ttl, True, self.instance),
      )
    return (Multiton.from_reduce_args, (self._factory, self._args, self._kw, self._ttl))

  def __hash__(self) -> int:
    return hash(self._key)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, Multiton):
      return NotImplemented
    return self._key == other._key

  @classmethod
  def _write_entry(cls, key: FrozenKey, obj: Any, ttl: float) -> None:
    """Write a cache entry, pushing a heap entry only for a finite TTL.
    Must be called under the lock.

    Eternal (``inf``) entries are never pushed: an ``inf`` deadline could
    never satisfy the sweep condition, so it would only bloat the heap."""
    seq = next(cls._SEQUENCE)
    now = time.monotonic()
    cls._INSTANCE_CACHE[key] = (obj, now, ttl, seq)

    if not math.isinf(ttl):
      heapq.heappush(cls._EXPIRY_HEAP, (now + ttl, seq, key))

  @classmethod
  def _compact_heap(cls) -> None:
    """Rebuild the heap from live, finite-TTL cache entries.
    Must be called under the lock.

    Iterating the cache and re-emitting one tuple per live finite-TTL key
    (with that entry's current seq) discards any finite tuple whose seq no
    longer matches its cache entry — i.e. stale entries left by TTL resets
    or releases that the sweep loop can't reach. Eternal (``inf``) entries
    are skipped, mirroring ``_write_entry`` never pushing them onto the heap.
    ``last + ttl`` reproduces the exact expiry pushed for that seq.
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
    entries are never on the heap, so it holds only finite tuples; once it
    grows much larger than the live cache it is compacted to reclaim the
    stale finite tuples left behind by TTL resets and releases.
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

    Cache hits hold only the brief global lock. On a miss the factory runs
    under a per-key lock — without the global lock — so a slow construction
    blocks only same-key callers, never access to other keys. The per-key
    lock is re-entrant: a factory may (transitively) access other multitons'
    ``instance``, as long as factory dependencies are acyclic.
    """
    with self._INSTANCE_LOCK:
      self._purge_expired()

      # Reset the TTL
      if (entry := self._INSTANCE_CACHE.get(self._key)) is not None:
        obj, _, ttl, _ = entry
        self._write_entry(self._key, obj, ttl)
        return obj

      if (key_lock := self._KEY_LOCKS.get(self._key)) is None:
        key_lock = self._KEY_LOCKS[self._key] = RLock()

    # Construction is serialised per key; the global lock is never held
    # across factory execution.
    with key_lock:
      # Double-check: another thread may have constructed this key while
      # we waited, and its entry may itself have expired in the meantime.
      with self._INSTANCE_LOCK:
        self._purge_expired()

        if (entry := self._INSTANCE_CACHE.get(self._key)) is not None:
          obj, _, ttl, _ = entry
          self._write_entry(self._key, obj, ttl)
          return obj

      obj = self._factory(*self._args, **self._kw)

      with self._INSTANCE_LOCK:
        self._write_entry(self._key, obj, self._ttl)
      return obj

  def release(self) -> None:
    """Immediately evict this Multiton's instance from the cache.

    Any Multiton sharing the same key will recreate the instance on next
    access. The corresponding heap entry is left in place and discarded
    as stale during the next purge sweep.

    A ``release()`` racing an in-flight construction of the same key does
    not wait for it: it evicts whatever entry currently exists (possibly
    none) and returns; the construction then publishes its entry as usual.
    """
    with self._INSTANCE_LOCK:
      self._INSTANCE_CACHE.pop(self._key, None)

  def __str__(self) -> str:
    return f"Multiton({self._factory})"

  __repr__ = __str__
