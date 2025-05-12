"""Sphinx configuration for PHASTA documentation."""

import os
import sys
from datetime import datetime

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "PHASTA Python"
copyright = f"{datetime.now().year}, PHASTA Team"
author = "PHASTA Team"

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "nbsphinx",
    "sphinx_rtd_theme",
]

# Extension settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Theme settings
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "style_nav_header_background": "#2980B9",
}

# HTML output settings
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_show_sphinx = False
html_show_copyright = True

# LaTeX output settings
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "figure_align": "htbp",
}

# PDF output settings
pdf_documents = [
    ("index", "phasta", "PHASTA Documentation", "PHASTA Team"),
]

# nbsphinx settings
nbsphinx_execute = "auto"
nbsphinx_allow_errors = True
nbsphinx_timeout = 600

# Todo settings
todo_include_todos = True
todo_emit_warnings = True

# Coverage settings
coverage_show_missing_items = True
coverage_statistics_to_report = True

# GitHub pages settings
github_url = "https://github.com/yourusername/phasta-py"
github_repo = "phasta-py"
github_branch = "main"
github_docs_path = "docs" 