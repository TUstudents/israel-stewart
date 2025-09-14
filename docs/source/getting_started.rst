Getting Started
===============

Installation
------------

Use the project's package manager to create the environment and install extras:

.. code-block:: bash

   uv sync --extra dev --extra docs

Quick Example
-------------

Run a benchmark to verify your setup:

.. code-block:: bash

   uv run python -m israel_stewart.benchmarks.bjorken_flow

Testing & Linting
-----------------

.. code-block:: bash

   uv run pytest -q
   uv run ruff check .
   uv run mypy israel_stewart
