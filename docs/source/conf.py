# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import axidence

project = "XENON axidence"
copyright = "2024, axidence contributors, the XENON collaboration"

release = axidence.__version__
version = axidence.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []  # type: ignore


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
# html_static_path = ['_static']

# Lets disable notebook execution for now
nbsphinx_allow_errors = True
nbsphinx_execute = "never"


def setup(app):
    # app.add_css_file('css/custom.css')
    # Hack to import something from this dir. Apparently we're in a weird
    # situation where you get a __name__  is not in globals KeyError
    # if you just try to do a relative import...
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    from build_release_notes import convert_release_notes

    this_dir = os.path.dirname(os.path.realpath(__file__))
    notes = os.path.join(this_dir, "..", "..", "HISTORY.md")
    os.makedirs(os.path.join(this_dir, "reference"), exist_ok=True)
    target = os.path.join(this_dir, "reference", "release_notes.rst")
    pull_url = "https://github.com/XENONnT/axidence/pull"

    convert_release_notes(notes, target, pull_url)
