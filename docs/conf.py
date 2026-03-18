# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup ---------------------------------------------------------------

# Add the project root to the path so that autodoc can find the modules.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information ------------------------------------------------------

project = "Special Relativity in Financial Modeling"
copyright = "2025-2026, SRFM Authors"
author = "SRFM Authors"
version = "0.1"
release = "0.1.0"

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output --------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "SRFM Documentation"

# -- autodoc options ----------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# -- Napoleon options (for Google/NumPy docstrings) ---------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- MyST parser options ------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
