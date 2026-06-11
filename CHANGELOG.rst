=========
Changelog
=========

All notable changes to ``rarg-python-patterns`` are documented in this file.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

Unreleased X.Y.Z (DD-MM-YYYY)
=============================

Added
-----

Changed
-------
- Rename the project from ``rarg-multiton`` to ``rarg-python-patterns`` and
  restructure it as a patterns collection: the Multiton now lives under the
  ``rarg_python_patterns.multiton`` subpackage. ``from rarg_python_patterns
  import Multiton`` continues to work via top-level re-exports (pr:`5`).

Fixed
-----

Removed
-------

0.0.1
=====

Added
-----
- Inline the deploy step in ``ci.yml`` (:pr:`4`)
- Support infinite TTL: ``with_ttl(math.inf)`` (and the ``with_infinite_ttl()``
  shorthand) makes a cache entry eternal — it never expires and is only removed
  by ``release()`` (:pr:`3`).
- Use ``rarg-gh-workflows`` to simplify and modularise Continuous Integration (:pr:`2`)
- Add ``register_freezer`` for custom type canonicalisation (:pr:`1`)
- Initial release: ``Multiton`` pattern with TTL-based cache expiry, extracted
  from `ratt-ru/xarray-kat`_.

.. _Keep a Changelog: https://keepachangelog.com/en/1.1.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html
.. _ratt-ru/xarray-kat: https://github.com/ratt-ru/xarray-kat
