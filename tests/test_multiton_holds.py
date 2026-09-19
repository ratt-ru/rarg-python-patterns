"""Tests for refcounted holds: ``acquire()``, ``release()`` and ``hold()``."""

import math
import pickle
import threading
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from rarg_python_patterns import Multiton

TIMEOUT = 10.0


@dataclass
class Data:
  a: float
  b: float


def test_multiton_acquire_returns_the_cached_instance():
  """acquire() creates the instance and hands back the cached object."""
  m = Multiton(Data, 1.0, b=3.0)
  obj = m.acquire()
  assert obj is m.instance
  assert m.hold_count == 1
  m.release()


def test_multiton_hold_survives_a_foreign_release():
  """A holder's entry is not evicted by another component's release()."""
  holder = Multiton(Data, 1.0, b=3.0)
  other = Multiton(Data, 1.0, 3.0)

  obj = holder.acquire()
  other.release()

  # The hold outlives the unrelated release, so the instance is unchanged
  assert other.instance is obj
  assert len(Multiton._INSTANCE_CACHE) == 1

  # ... and the last holder letting go does evict it
  holder.release()
  assert len(Multiton._INSTANCE_CACHE) == 0
  assert other.instance is not obj


def test_multiton_holds_are_refcounted():
  """Only the last of several holds evicts the entry."""
  m = Multiton(Data, 1.0, b=3.0)
  obj = m.acquire()
  assert m.acquire() is obj
  assert m.acquire() is obj
  assert m.hold_count == 3

  for expected in (2, 1):
    m.release()
    assert m.hold_count == expected
    assert m.instance is obj

  m.release()
  assert m.hold_count == 0
  assert len(Multiton._INSTANCE_CACHE) == 0


def test_multiton_hold_count_is_shared_by_equal_multitons():
  """Holds are counted per key, not per Multiton object."""
  m1 = Multiton(Data, 1.0, b=3.0)
  m2 = Multiton(Data, 1.0, 3.0)

  obj = m1.acquire()
  assert m2.hold_count == 1

  m2.acquire()
  assert m1.hold_count == 2

  # Each holder must release; the first release leaves the entry in place
  m1.release()
  assert m2.instance is obj
  m2.release()
  assert len(Multiton._INSTANCE_CACHE) == 0


def test_multiton_held_entry_does_not_expire():
  """A held entry survives an arbitrarily large clock advance."""
  t = [0.0]

  with patch("rarg_python_patterns.multiton.multiton.time") as mock_time:
    mock_time.monotonic.side_effect = lambda: t[0]

    m = Multiton(Data, 1.0, b=3.0).with_args(ttl=10.0)
    obj = m.acquire()

    t[0] = 1e6
    assert m.instance is obj

    # Releasing the last hold evicts, and the TTL governs the next entry
    m.release()
    new_obj = m.instance
    assert new_obj is not obj

    t[0] += 20.0
    assert m.instance is not new_obj


def test_multiton_held_entry_is_not_pushed_onto_the_heap():
  """Pinning an entry makes it eternal, and eternal entries skip the heap."""
  m = Multiton(Data, 1.0, b=3.0).with_args(ttl=10.0)
  m.instance
  assert len(Multiton._EXPIRY_HEAP) == 1

  m.acquire()
  _, _, ttl, seq = Multiton._INSTANCE_CACHE[m._key]
  assert math.isinf(ttl)

  # The stale finite tuple left behind names a superseded seq, so the sweep
  # discards it rather than evicting the held entry
  assert all(heap_seq != seq for _, heap_seq, _ in Multiton._EXPIRY_HEAP)
  m.release()


def test_multiton_release_without_a_hold_evicts_immediately():
  """The pre-holds contract: an unheld release() evicts there and then."""
  m = Multiton(Data, 1.0, b=3.0)
  obj = m.instance
  assert len(Multiton._INSTANCE_CACHE) == 1

  m.release()
  assert len(Multiton._INSTANCE_CACHE) == 0
  assert m.instance is not obj
  m.release()


def test_multiton_unbalanced_release_is_harmless():
  """Releasing more often than acquiring neither raises nor goes negative."""
  m = Multiton(Data, 1.0, b=3.0)
  m.acquire()
  m.release()
  m.release()
  assert m.hold_count == 0


def test_multiton_hold_context_manager():
  """hold() yields the instance and releases it on exit."""
  m = Multiton(Data, 1.0, b=3.0)

  with m.hold() as obj:
    assert obj is m.instance
    assert m.hold_count == 1

  assert m.hold_count == 0
  assert len(Multiton._INSTANCE_CACHE) == 0


def test_multiton_hold_context_manager_releases_on_exception():
  """An exception inside the block still drops the hold."""
  m = Multiton(Data, 1.0, b=3.0)

  with pytest.raises(RuntimeError, match="boom"):
    with m.hold():
      assert m.hold_count == 1
      raise RuntimeError("boom")

  assert m.hold_count == 0


def test_multiton_nested_holds_of_the_same_key():
  """Nested hold() blocks see one instance, evicted when the outer exits."""
  m = Multiton(Data, 1.0, b=3.0)

  with m.hold() as outer:
    with m.hold() as inner:
      assert inner is outer
    # The inner block exiting must not evict the outer block's instance
    assert m.instance is outer

  assert len(Multiton._INSTANCE_CACHE) == 0


def test_multiton_clear_cache_drops_holds():
  """clear_cache() discards instances, holds and heap entries alike."""
  m = Multiton(Data, 1.0, b=3.0)
  m.acquire()
  assert m.hold_count == 1

  Multiton.clear_cache()
  assert m.hold_count == 0
  assert len(Multiton._INSTANCE_CACHE) == 0
  assert len(Multiton._EXPIRY_HEAP) == 0


def test_multiton_concurrent_holds_keep_one_instance_alive():
  """Under concurrent acquire/release the entry lives until the last release.

  Every thread holds the key for a while, releasing only at the end, so no
  thread may observe a different instance from any other.
  """
  nthreads = 8
  calls = []
  calls_lock = threading.Lock()
  start = threading.Barrier(nthreads, timeout=TIMEOUT)
  acquired = threading.Barrier(nthreads, timeout=TIMEOUT)
  seen = []

  def factory(value: float) -> Data:
    with calls_lock:
      calls.append(value)
    return Data(value, value)

  def worker():
    m = Multiton(factory, 1.0)
    start.wait()
    obj = m.acquire()
    # Nobody releases before everybody has acquired, so the entry is held
    # continuously and these accesses must all return the same object.
    acquired.wait()
    seen.append((obj, m.instance))
    m.release()

  threads = [threading.Thread(target=worker) for _ in range(nthreads)]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join(TIMEOUT)
    assert not thread.is_alive(), "thread deadlocked or timed out"

  assert len(calls) == 1, "the factory ran more than once"
  assert len(seen) == nthreads
  first = seen[0][0]
  assert all(obj is first and inst is first for obj, inst in seen)

  # The last release evicted the entry
  assert len(Multiton._INSTANCE_CACHE) == 0
  assert Multiton(factory, 1.0).hold_count == 0


def test_multiton_release_from_a_non_holder_is_a_noop_while_held():
  """An eviction request loses to an outstanding hold, and is not a hold."""
  holder = Multiton(Data, 1.0, b=3.0)
  other = Multiton(Data, 1.0, 3.0)

  obj = holder.acquire()
  other.release()
  other.release()

  # Neither release consumed the hold
  assert holder.hold_count == 1
  assert holder.instance is obj

  holder.release()
  assert len(Multiton._INSTANCE_CACHE) == 0


def test_multiton_unpickled_multiton_holds_nothing():
  """Holds are per-process state and do not travel through a pickle."""
  m = Multiton(Data, 1.0, b=3.0)
  obj = m.acquire()

  unpickled = pickle.loads(pickle.dumps(m))
  assert unpickled.instance is obj

  # The unpickled Multiton is not a holder, so it cannot evict the entry
  unpickled.release()
  assert m.hold_count == 1
  assert m.instance is obj

  m.release()
  assert len(Multiton._INSTANCE_CACHE) == 0
