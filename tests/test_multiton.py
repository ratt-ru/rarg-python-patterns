import math
import pickle
from dataclasses import dataclass
from unittest.mock import patch

import cloudpickle
import dill
import pytest

from rarg_python_patterns import Multiton


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


class DefaultedData:
  def __init__(self, a: float, b: float = 10.0):
    self.a, self.b = a, b


def test_multiton_class_factory_defaults():
  """A class factory with a defaulted __init__ normalises correctly.

  Before the fix, getfullargspec's parameter list included ``self``,
  shifting positional matching by one: fully-positional calls gained a
  duplicated default (TypeError at .instance) and equivalent calls
  produced distinct cache keys.
  """
  m1 = Multiton(DefaultedData, 1.0)
  m2 = Multiton(DefaultedData, 1.0, 10.0)
  m3 = Multiton(DefaultedData, 1.0, b=10.0)
  assert m1 == m2 == m3
  assert m1.instance is m2.instance is m3.instance
  assert (m1.instance.a, m1.instance.b) == (1.0, 10.0)


def test_multiton_class_factory_all_defaulted():
  """A single positional arg to a fully-defaulted __init__ works."""

  class AllDefaulted:
    def __init__(self, a=10, b=20):
      self.a, self.b = a, b

  m = Multiton(AllDefaulted, 1)
  assert (m.instance.a, m.instance.b) == (1, 20)


def test_multiton_class_factory_invalid_call_raises_at_construction():
  """Invalid factory arguments fail at Multiton(...) time, not .instance."""
  with pytest.raises(TypeError):
    Multiton(DefaultedData, 1.0, 2.0, 3.0)


def test_multiton_class_factory_default_pickle():
  """A class factory with defaults round-trips through pickle.

  Re-normalisation on reconstruction must be idempotent: the unpickled
  Multiton produces the same key and a valid call.
  """
  m = Multiton(DefaultedData, 1.0)
  m2 = pickle.loads(pickle.dumps(m))
  assert m == m2
  assert m.instance is m2.instance
  assert (m2.instance.a, m2.instance.b) == (1.0, 10.0)


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

  with patch("rarg_python_patterns.multiton.multiton.time") as mock_time:
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

  with patch("rarg_python_patterns.multiton.multiton.time") as mock_time:
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


def test_multiton_infinite_ttl_never_expires():
  """An eternal entry survives an arbitrarily large clock advance."""
  t = [0.0]

  def fake_monotonic():
    return t[0]

  with patch("rarg_python_patterns.multiton.multiton.time") as mock_time:
    mock_time.monotonic.side_effect = fake_monotonic

    m = Multiton(Data, 1.0, b=3.0).with_infinite_ttl()
    inst1 = m.instance

    t[0] = 1e18
    assert m.instance is inst1


def test_multiton_with_ttl_inf_equivalent():
  """with_ttl(math.inf) is eternal; with_ttl(x) matches with_args(ttl=x)."""
  m_inf = Multiton(Data, 1.0, b=3.0).with_ttl(math.inf)
  assert m_inf._ttl == math.inf

  m_finite = Multiton(Data, 2.0, b=3.0).with_ttl(10.0)
  m_args = Multiton(Data, 2.0, b=3.0).with_args(ttl=10.0)
  assert m_finite._ttl == m_args._ttl == 10.0


def test_multiton_eternal_ttl_never_grows_heap():
  """An eternal entry is never pushed onto the heap, regardless of access count.

  An ``inf`` deadline could never satisfy the sweep condition, so _write_entry
  skips the heap push entirely for eternal entries. Creating and repeatedly
  accessing (each access a TTL reset, i.e. another _write_entry) a lone eternal
  entry must therefore leave the heap empty the whole time — no orphaned inf
  tuples accumulate and no compaction is ever needed to reclaim them.
  """
  m = Multiton(Data, 1.0, b=3.0).with_infinite_ttl()
  inst1 = m.instance
  assert len(Multiton._EXPIRY_HEAP) == 0
  for _ in range(50):
    assert m.instance is inst1
    assert len(Multiton._EXPIRY_HEAP) == 0


def test_multiton_finite_heap_compaction_discards_stale():
  """Compaction drops stale (seq-mismatched) finite tuples, keeping the live one."""
  t = [0.0]

  def fake_monotonic():
    return t[0]

  with (
    patch("rarg_python_patterns.multiton.multiton.time") as mock_time,
    patch.object(Multiton, "_HEAP_COMPACT_MIN", 4),
    patch.object(Multiton, "_HEAP_COMPACT_FACTOR", 2.0),
  ):
    mock_time.monotonic.side_effect = fake_monotonic

    m = Multiton(Data, 1.0, b=3.0).with_args(ttl=1000.0)
    inst1 = m.instance
    # Reset the TTL repeatedly without advancing past it: each access orphans a
    # heap tuple with a stale seq. None are popped (deadline far in the future),
    # so without compaction the heap would grow to ~50. Compaction discards the
    # stale tuples on the seq mismatch, keeping the heap bounded and the single
    # live entry intact.
    for _ in range(50):
      assert m.instance is inst1
      assert len(Multiton._EXPIRY_HEAP) <= Multiton._HEAP_COMPACT_MIN + 1
    assert len(Multiton._INSTANCE_CACHE) == 1


def test_multiton_nan_ttl_rejected():
  """A nan TTL is rejected; it would never expire and would leak in the heap."""
  with pytest.raises(ValueError, match="real number"):
    Multiton(Data, 1.0, b=3.0).with_args(ttl=math.nan)
  with pytest.raises(ValueError, match="real number"):
    Multiton(Data, 1.0, b=3.0).with_ttl(math.nan)


def test_multiton_non_numeric_ttl_rejected():
  """A non-real TTL is rejected with a ValueError, not a downstream TypeError."""
  for bad in (None, "10", 1j):
    with pytest.raises(ValueError, match="real number"):
      Multiton(Data, 1.0, b=3.0).with_args(ttl=bad)


def test_multiton_int_ttl_accepted():
  """An int TTL is a valid real number and behaves like its float value."""
  m = Multiton(Data, 1.0, b=3.0).with_ttl(10)
  assert m._ttl == 10
  assert m.instance == Data(1.0, 3.0)


def test_multiton_infinite_ttl_pickle_roundtrip():
  """An infinite TTL is preserved through a pickle round-trip."""
  m = Multiton(Data, 1.0, b=3.0).with_infinite_ttl()
  m2 = pickle.loads(pickle.dumps(m))
  assert m2._ttl == math.inf


def test_multiton_mixed_finite_and_eternal():
  """A finite entry expires and is swept while an eternal entry survives."""
  t = [0.0]

  def fake_monotonic():
    return t[0]

  with patch("rarg_python_patterns.multiton.multiton.time") as mock_time:
    mock_time.monotonic.side_effect = fake_monotonic

    finite = Multiton(Data, 1.0, b=1.0).with_args(ttl=5.0)
    eternal = Multiton(Data, 2.0, b=2.0).with_infinite_ttl()
    finite.instance
    eternal_inst = eternal.instance
    assert len(Multiton._INSTANCE_CACHE) == 2
    # Only the finite entry is on the heap; the eternal one is never pushed.
    assert len(Multiton._EXPIRY_HEAP) == 1

    # Advance past the finite TTL; accessing the eternal entry triggers a sweep.
    t[0] = 10.0
    assert eternal.instance is eternal_inst
    assert finite._key not in Multiton._INSTANCE_CACHE
    assert eternal._key in Multiton._INSTANCE_CACHE


def test_multiton_infinite_ttl_release():
  """An eternal entry can be released and is recreated on next access."""
  m = Multiton(Data, 1.0, b=3.0).with_infinite_ttl()
  inst1 = m.instance
  assert len(Multiton._INSTANCE_CACHE) == 1

  m.release()
  assert len(Multiton._INSTANCE_CACHE) == 0

  inst2 = m.instance
  assert inst2 is not inst1
  assert inst2 == inst1


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

  with patch("rarg_python_patterns.multiton.multiton.time") as mock_time:
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

  with patch("rarg_python_patterns.multiton.multiton.time") as mock_time:
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
