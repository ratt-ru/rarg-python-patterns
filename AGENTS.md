# rarg-python-patterns: a library of Python software patterns

[ratt-ru/rarg-python-patterns](https://github.com/ratt-ru/rarg-python-patterns) is a
small Python library of reusable software patterns. It currently provides a
`Multiton` (see the [Multiton pattern](https://en.wikipedia.org/wiki/Multiton_pattern))
with TTL-based cache expiry.

## Layout

- [`multiton/multiton.py`](src/rarg_python_patterns/multiton/multiton.py): the
  `Multiton` class. Its docstring documents TTLs, eternal entries, instance
  serialisation and thread safety.
- [`multiton/canonicalisation.py`](src/rarg_python_patterns/multiton/canonicalisation.py):
  `FrozenKey`, `freeze` and `normalise_args`, which turn factory arguments into
  hashable cache keys. Register freezers for new types with `register_freezer`.

## Conventions

- No required runtime dependencies. `numpy` is optional: code that uses it must
  still import and work without it.
- `Multiton` must stay hashable, equality-comparable and pickleable (stdlib
  `pickle`, `cloudpickle` and `dill`). Keep the round-trip tests passing.
- ruff for formatting and linting: **2-space indent**, line length 88, imports
  sorted by the ruff `I` rules.
- pytest: one behaviour per test, named for the behaviour; prefer fixtures over
  setup/teardown; keep tests deterministic.

## Testing notes

The tests need no external data; the `test` dependency group installs numpy,
cloudpickle and dill.

- `Multiton`'s cache is class-level state shared by every instance. The autouse
  `clear_multitons` fixture in [`tests/conftest.py`](tests/conftest.py) clears it
  around each test. If you add class-level state, clear it there too.

## Tooling

Use `uv` to interact with this package.

- Install: `uv sync`
- Test: `uv run --group test pytest tests/`
- Pre-commit hooks (ruff, mypy and file hygiene) gate commits; run them before
  proposing a change as done: `uv run --group dev pre-commit run -a`
- Versioning: tbump (never hand-edit version strings): `uv run --group dev tbump <new-version>`.

Update [`CHANGELOG.rst`](CHANGELOG.rst) for user-visible changes.
