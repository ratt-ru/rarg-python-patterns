import pickle
from dataclasses import dataclass
from unittest.mock import patch

import cloudpickle
import dill
import pytest

from rarg_multiton import Multiton


@dataclass
class Data:
  a: float
  b: float


class DataFactory:
  @classmethod
  def create(cls, a: float, b: float = 3.0) -> Data:
    return Data(a, b)


def test_multiton_arg_normalisation():
  """Test that factory keywords are correctly normalised into args"""
  m1 = Multiton(Data, 2.0, b=3.0)
  m2 = Multiton(Data, 2.0, 3.0)
  assert m1.instance is m2.instance


@pytest.mark.parametrize("method", [pickle, cloudpickle, dill])
def test_multiton_pickle(method):
  """Tests multiton pickling with difference pickle implementations"""
  m = Multiton(Data, 2.0, b=3.0)
  datum = {"d": m, "e": {"f": m}}
  udatum = method.loads(method.dumps(datum))
  assert datum["d"].instance is datum["e"]["f"].instance
  assert udatum["d"].instance is udatum["e"]["f"].instance
  assert datum["d"].instance is udatum["d"].instance


def test_multiton_release():
  """Tests that release() immediately evicts the entry from the shared cache."""
  m1 = Multiton(Data, 1.0, b=3.0)
  m2 = Multiton(Data, 1.0, 3.0)
  obj = m1.instance
  assert m1.instance is m2.instance
  assert len(Multiton._INSTANCE_CACHE) == 1

  # Release via m2 evicts the entry for all Multitons with this key
  m2.release()
  assert len(Multiton._INSTANCE_CACHE) == 0

  # m1.instance creates a fresh instance now
  new_inst = m1.instance
  assert new_inst is not obj
  assert len(Multiton._INSTANCE_CACHE) == 1

  m1.release()
  assert len(Multiton._INSTANCE_CACHE) == 0


def test_multiton_reentrant():
  """Tests RLock works"""

  def inner_factory(m: Multiton[Data]) -> Data:
    return m.instance

  def outer_factory(a: int, m: Multiton[Data]) -> Data:
    return inner_factory(m)

  om = Multiton(outer_factory, 2, Multiton(Data, 1.0, b=2.0))
  assert om.instance.a == 1.0
  assert om.instance.b == 2.0


def test_multiton_classmethod_normalisation():
  """Test that normalise_args correctly handles classmethods (skips bound cls)"""
  m1 = Multiton(DataFactory.create, 2.0, b=3.0)
  m2 = Multiton(DataFactory.create, 2.0, 3.0)
  assert m1.instance is m2.instance


def test_multiton_classmethod_pickle():
  """Test that a Multiton with a classmethod factory round-trips through pickle.

  This exercises the normalise_args fix: after unpickling, the reconstructed
  Multiton must produce the same key and call the factory with the same args
  as the original, without cls being counted as a positional slot.

  stdlib pickle serialises classmethods by reference, so the deserialized key
  is identical and we get cache sharing (m.instance is m2.instance).
  cloudpickle/dill may reconstruct a new bound-method object whose hash differs,
  so we only assert value equality for those.
  """
  m = Multiton(DataFactory.create, 2.0, b=3.0)
  m2 = pickle.loads(pickle.dumps(m))
  # stdlib pickle: key must be identical → cache hit → same object
  assert m.instance is m2.instance
  assert m2.instance == Data(2.0, 3.0)

  for method in [cloudpickle, dill]:
    m3 = method.loads(method.dumps(m))
    # cloudpickle/dill may not preserve bound-method identity across streams,
    # but the factory must still be callable with the correct arguments.
    assert m3.instance == Data(2.0, 3.0)


def test_multiton_classmethod_default_not_duplicated():
  """Test that a defaulted kwarg isn't appended twice when unpickling.

  Before the fix, the chunk_store-style bug would cause normalise_args to
  append the default value again on reconstruction, producing a key mismatch
  and a TypeError when calling the factory with too many positional args.
  """
  m = Multiton(DataFactory.create, 2.0, b=5.0)
  m2 = pickle.loads(pickle.dumps(m))
  assert m == m2
  assert m2.instance == Data(2.0, 5.0)


def test_multiton_cache_shared_on_first_access():
  """Tests that a second Multiton picks up an already-cached instance."""
  m1 = Multiton(Data, 1.0, b=3.0)
  assert m1.instance == Data(1.0, 3.0)

  m2 = Multiton(Data, 1.0, b=3.0)
  assert m2.instance is m1.instance


def test_multiton_ttl_expiry():
  """Instance is recreated after TTL expires."""
  t = [0.0]

  def fake_monotonic():
    return t[0]

  with patch("rarg_multiton.multiton.time") as mock_time:
    mock_time.monotonic.side_effect = fake_monotonic

    m = Multiton(Data, 1.0, b=3.0).with_args(ttl=10.0)
    inst1 = m.instance

    # Still within TTL — same instance
    t[0] = 9.0
    assert m.instance is inst1

    # Past TTL — new instance
    t[0] = 20.0
    inst2 = m.instance
    assert inst2 is not inst1
    assert inst2 == inst1


def test_multiton_ttl_reset_on_access():
  """Accessing an instance resets its TTL."""
  t = [0.0]

  def fake_monotonic():
    return t[0]

  with patch("rarg_multiton.multiton.time") as mock_time:
    mock_time.monotonic.side_effect = fake_monotonic

    m = Multiton(Data, 1.0, b=3.0).with_args(ttl=10.0)
    inst1 = m.instance  # created at t=0, last_access=0

    # Access at t=9 resets last_access to 9
    t[0] = 9.0
    assert m.instance is inst1

    # At t=18 only 9s have elapsed since last access — still alive
    t[0] = 18.0
    assert m.instance is inst1

    # At t=29 more than 10s have elapsed since last access at t=18
    t[0] = 29.0
    inst2 = m.instance
    assert inst2 is not inst1


def test_multiton_ttl_pickle_roundtrip():
  """TTL is preserved through pickle round-trip."""
  m = Multiton(Data, 1.0, b=3.0).with_args(ttl=42.0)
  m2 = pickle.loads(pickle.dumps(m))
  assert m2._ttl == 42.0


def test_multiton_default_ttl():
  """Omitting ttl uses _DEFAULT_TTL."""
  m = Multiton(Data, 1.0, b=3.0)
  assert m._ttl == Multiton._DEFAULT_TTL


def test_multiton_expired_entries_swept_on_access():
  """Expired entries from other keys are removed when any instance is accessed."""
  t = [0.0]

  def fake_monotonic():
    return t[0]

  with patch("rarg_multiton.multiton.time") as mock_time:
    mock_time.monotonic.side_effect = fake_monotonic

    m1 = Multiton(Data, 1.0, b=1.0).with_args(ttl=5.0)
    m2 = Multiton(Data, 2.0, b=2.0).with_args(ttl=100.0)
    m1.instance
    m2.instance
    assert len(Multiton._INSTANCE_CACHE) == 2

    # Advance past m1's TTL; access m2 to trigger sweep
    t[0] = 10.0
    m2.instance
    assert len(Multiton._INSTANCE_CACHE) == 1
    assert m1._key not in Multiton._INSTANCE_CACHE


def test_multiton_heap_stale_entries_discarded():
  """TTL resets push new heap entries; stale entries are discarded during purge."""
  t = [0.0]

  def fake_monotonic():
    return t[0]

  with patch("rarg_multiton.multiton.time") as mock_time:
    mock_time.monotonic.side_effect = fake_monotonic

    m = Multiton(Data, 1.0, b=3.0).with_args(ttl=10.0)
    m.instance  # heap: 1 entry (expiry=10)
    assert len(Multiton._EXPIRY_HEAP) == 1

    t[0] = 5.0
    m.instance  # TTL reset: heap grows to 2 (old stale + new at expiry=15)
    assert len(Multiton._EXPIRY_HEAP) == 2

    # At t=12 the stale entry (expiry=10) is past its deadline but the live
    # entry (expiry=15) is not. Purge should discard the stale entry only.
    t[0] = 12.0
    m.instance  # triggers purge, pops stale entry; live entry refreshed
    assert len(Multiton._INSTANCE_CACHE) == 1

    # Release removes from cache; orphaned heap entry is discarded during next purge.
    m.release()
    assert len(Multiton._INSTANCE_CACHE) == 0
    m.instance  # recreates; purge discards the orphaned heap entry(s)
    assert len(Multiton._INSTANCE_CACHE) == 1
