# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # this allows sphinx to find gpml/

project = 'gpml'
copyright = '2025, Majed Jaber'
author = 'Majed Jaber'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme' #alabaster
html_static_path = ['_static']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # if using Google/NumPy-style docstrings
    'sphinx.ext.viewcode',
]
