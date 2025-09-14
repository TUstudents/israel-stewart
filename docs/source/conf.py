# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure package import works for autodoc
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Read project metadata from pyproject.toml
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore

project = "israel-stewart"
author = "Relativistic Hydrodynamics Team"
release = ""
pyproject = ROOT / "pyproject.toml"
if tomllib and pyproject.exists():
    data = tomllib.loads(pyproject.read_text())
    meta = data.get("project", {})
    project = meta.get("name", project)
    release = meta.get("version", "")
    authors = meta.get("authors", [])
    if authors:
        # join all author names if present
        names = [a.get("name") or a.get("email") for a in authors if isinstance(a, dict)]
        author = ", ".join([n for n in names if n]) or author

# Sphinx requires a string for copyright
copyright = f"2025, {author}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.duration",
    "nbsphinx",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
