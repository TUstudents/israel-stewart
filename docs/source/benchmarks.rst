Benchmarks
==========

This project includes reference problems under ``israel_stewart/benchmarks``.

Available Benchmarks
--------------------

- ``bjorken_flow.py`` — 1D boost-invariant expansion.
- ``equilibration.py`` — relaxation toward equilibrium.
- ``sound_waves.py`` — linear wave propagation.

Running
-------

Use ``uv run`` to execute modules directly:

.. code-block:: bash

   uv run python -m israel_stewart.benchmarks.bjorken_flow

Outputs may include console diagnostics and plots. Prefer saving figures via
``israel_stewart.utils.visualization`` and data via ``israel_stewart.utils.io``.

