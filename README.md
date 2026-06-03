# rarg-multiton

A small, self-contained implementation of the
[Multiton pattern](https://en.wikipedia.org/wiki/Multiton_pattern) with
TTL-based cache expiry.

A `Multiton` wraps a factory function and its arguments. The underlying
instance is created lazily on first access to `.instance` and cached, keyed on
the (frozen, hashable) factory and arguments. Any `Multiton` constructed with
the same factory and arguments shares the cached instance. `Multiton` objects
are themselves hashable, equality-comparable and pickleable (with `pickle`,
`cloudpickle` and `dill`), provided their arguments are too.

Cached instances expire after `ttl` seconds of inactivity; accessing
`.instance` resets the TTL. Expired entries are swept on every access via a
min-heap ordered by expiry time.

```python
from rarg_multiton import Multiton


def open_connection(url: str, timeout: float = 1.0) -> Connection:
    ...


# The resource is only created when `.instance` is first accessed.
resource = Multiton(open_connection, "https://www.python.org", timeout=10.0)
response = resource.instance.request("GET", "/foo/bar.html")

# Override the default TTL (300s) per-instance.
resource = Multiton(open_connection, "https://www.python.org").with_args(ttl=60.0)
```

## Installation

```bash
pip install rarg-multiton
```

The package has **no required runtime dependencies**. `numpy` is an optional
dependency: when installed, `numpy.ndarray` factory arguments are frozen by
their bytes, shape and dtype so they can be used as cache keys.

## Development

This package uses [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run --group test py.test tests/
uv run --dev pre-commit run -a
```
