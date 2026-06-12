"""Concurrency tests for per-key construction locking.

Every test forces an interesting interleaving with :class:`threading.Event`,
but all assertions are written to hold under *any* interleaving — the events
steer the schedule, they never decide pass/fail.
"""

import gc
import math
import random
import threading

import pytest

from rarg_python_patterns import Multiton

TIMEOUT = 10.0


def join_or_fail(thread: threading.Thread, timeout: float = TIMEOUT) -> None:
  """Join a thread, failing the test (rather than hanging) on deadlock."""
  thread.join(timeout)
  assert not thread.is_alive(), "thread deadlocked or timed out"


class Item:
  """Unique-per-construction payload for identity assertions."""

  def __init__(self, value: float):
    self.value = value


class GatedFactory:
  """Factory that blocks inside construction until the test releases it.

  ``entered`` is set when a call reaches the gate, letting tests act while
  a construction is verifiably in flight. Only the first call gates;
  subsequent calls (retries, distinct keys) return immediately.
  """

  def __init__(self, fail_first: bool = False):
    self.entered = threading.Event()
    self.gate = threading.Event()
    self.fail_first = fail_first
    self.calls = 0
    self._calls_lock = threading.Lock()

  def __call__(self, value: float = 0.0) -> Item:
    with self._calls_lock:
      self.calls += 1
      first = self.calls == 1
    if first:
      self.entered.set()
      assert self.gate.wait(TIMEOUT)
      if self.fail_first:
        raise RuntimeError("first construction fails")
    return Item(value)


def test_multiton_slow_factory_does_not_block_other_keys():
  """The headline fix: a construction in flight on one key must not stall
  cache hits or fresh constructions on other keys. Hangs under a global
  lock held across factory execution."""
  factory = GatedFactory()
  slow = Multiton(factory)

  fast_hit = Multiton(Item, 1.0)
  fast_hit.instance  # pre-populate: exercises the hit path below
  fast_miss = Multiton(Item, 2.0)  # exercises the construction path below

  t = threading.Thread(target=lambda: slow.instance)
  t.start()
  assert factory.entered.wait(TIMEOUT)  # slow construction now in flight

  results = []
  other = threading.Thread(
    target=lambda: results.extend([fast_hit.instance, fast_miss.instance])
  )
  other.start()
  join_or_fail(other, timeout=5.0)
  assert len(results) == 2

  factory.gate.set()
  join_or_fail(t)
  assert isinstance(slow.instance, Item)


def test_multiton_global_lock_free_and_key_lock_present_during_construction():
  """While a factory runs, the global lock must be available to other
  threads and exactly one per-key lock must be registered."""
  factory = GatedFactory()
  m = Multiton(factory)

  t = threading.Thread(target=lambda: m.instance)
  t.start()
  assert factory.entered.wait(TIMEOUT)

  assert Multiton._INSTANCE_LOCK.acquire(timeout=5.0), (
    "global lock held during factory execution"
  )
  try:
    assert len(Multiton._KEY_LOCKS) == 1
  finally:
    Multiton._INSTANCE_LOCK.release()

  factory.gate.set()
  join_or_fail(t)


def test_multiton_same_key_race_constructs_once():
  """N threads racing on the same missing key run the factory exactly once
  and all receive the identical object."""
  factory = GatedFactory()
  n_threads = 8
  barrier = threading.Barrier(n_threads)
  results: list = [None] * n_threads

  def worker(i: int) -> None:
    barrier.wait()
    results[i] = Multiton(factory).instance

  threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
  for t in threads:
    t.start()
  assert factory.entered.wait(TIMEOUT)
  factory.gate.set()
  for t in threads:
    join_or_fail(t)

  assert factory.calls == 1
  assert all(r is results[0] for r in results)


def test_multiton_factory_exception_leaves_no_entry_and_no_lock():
  """A raising factory writes no cache entry and leaks no key lock; the
  next access retries the factory and succeeds."""
  factory = GatedFactory(fail_first=True)
  factory.gate.set()  # don't block; we only want the first-call failure
  m = Multiton(factory)

  with pytest.raises(RuntimeError, match="first construction fails"):
    m.instance
  assert len(Multiton._INSTANCE_CACHE) == 0
  gc.collect()
  assert len(Multiton._KEY_LOCKS) == 0

  assert isinstance(m.instance, Item)
  assert factory.calls == 2


def test_multiton_factory_exception_waiter_retries():
  """A waiter blocked on the key lock while the winner's factory raises
  must run the factory itself and succeed, not deadlock or see the error."""
  factory = GatedFactory(fail_first=True)
  errors: list = []
  results: list = []

  def winner() -> None:
    try:
      Multiton(factory).instance
    except RuntimeError as e:
      errors.append(e)

  def waiter() -> None:
    results.append(Multiton(factory).instance)

  t1 = threading.Thread(target=winner)
  t1.start()
  assert factory.entered.wait(TIMEOUT)
  t2 = threading.Thread(target=waiter)
  t2.start()
  factory.gate.set()
  join_or_fail(t1)
  join_or_fail(t2)

  assert len(errors) == 1
  assert factory.calls == 2
  assert isinstance(results[0], Item)


def test_multiton_waiter_does_not_resurrect_expired_entry():
  """The double-check after acquiring the key lock must re-purge: an entry
  that expired while the waiter was blocked is reconstructed, not returned
  with a refreshed TTL. A ttl of 0 makes the winner's entry
  expired-on-arrival, so any interleaving must construct twice."""
  factory = GatedFactory()
  results: dict = {}

  def access(name: str) -> None:
    results[name] = Multiton(factory).with_ttl(0.0).instance

  t1 = threading.Thread(target=access, args=("winner",))
  t1.start()
  assert factory.entered.wait(TIMEOUT)
  t2 = threading.Thread(target=access, args=("waiter",))
  t2.start()
  factory.gate.set()
  join_or_fail(t1)
  join_or_fail(t2)

  assert factory.calls == 2
  assert results["winner"] is not results["waiter"]


def test_multiton_nested_factory_cross_key_under_contention():
  """A factory resolving another multiton's instance (holding its own key
  lock) must not deadlock against threads hammering the inner key."""
  inner = Multiton(Item, 1.0)
  entered, gate = threading.Event(), threading.Event()

  def outer_factory(m: Multiton) -> Item:
    entered.set()
    assert gate.wait(TIMEOUT)
    return m.instance  # acquires inner key lock while holding outer's

  outer = Multiton(outer_factory, inner)
  t1 = threading.Thread(target=lambda: outer.instance)
  t1.start()
  assert entered.wait(TIMEOUT)

  t2 = threading.Thread(target=lambda: [inner.instance for _ in range(100)])
  t2.start()
  gate.set()
  join_or_fail(t1)
  join_or_fail(t2)

  assert outer.instance is inner.instance


def test_multiton_key_locks_do_not_leak():
  """Key locks vanish once no thread is constructing or waiting, even
  though the cache entry itself lives on."""
  factory = GatedFactory()
  m = Multiton(factory)

  t = threading.Thread(target=lambda: m.instance)
  t.start()
  assert factory.entered.wait(TIMEOUT)
  assert len(Multiton._KEY_LOCKS) == 1

  factory.gate.set()
  join_or_fail(t)
  gc.collect()
  assert len(Multiton._KEY_LOCKS) == 0
  assert len(Multiton._INSTANCE_CACHE) == 1


def test_multiton_release_during_construction():
  """release() racing an in-flight same-key construction returns promptly
  as a no-op; the construction then publishes its entry as documented."""
  factory = GatedFactory()
  m = Multiton(factory)

  t = threading.Thread(target=lambda: m.instance)
  t.start()
  assert factory.entered.wait(TIMEOUT)

  releaser = threading.Thread(target=m.release)
  releaser.start()
  join_or_fail(releaser, timeout=5.0)  # must not block on the key lock

  factory.gate.set()
  join_or_fail(t)
  assert len(Multiton._INSTANCE_CACHE) == 1


def test_multiton_concurrent_soak():
  """Schedule-noise smoke test: many threads churning instance/release
  across shared keys and TTLs must neither deadlock nor corrupt state."""
  n_threads, n_keys, iters = 8, 4, 200
  errors: list = []

  def factory(k: int) -> Item:
    return Item(float(k))

  def worker(seed: int) -> None:
    rng = random.Random(seed)
    try:
      for _ in range(iters):
        k = rng.randrange(n_keys)
        m = Multiton(factory, k).with_ttl(rng.choice([0.001, 0.05, math.inf]))
        if rng.random() < 0.1:
          m.release()
        else:
          assert m.instance.value == float(k)
    except Exception as e:  # pragma: no cover - failure path
      errors.append(e)

  threads = [threading.Thread(target=worker, args=(s,)) for s in range(n_threads)]
  for t in threads:
    t.start()
  for t in threads:
    join_or_fail(t, timeout=30.0)
  assert not errors
