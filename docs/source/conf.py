# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
project = 'Ultrasound Imaging Simulator'
copyright = '2024, Parham Baghbanbashi, Luca Hutchison, Amar Saed, Meriyam Sena, Yimin Xu'
author = 'Parham Baghbanbashi, Luca Hutchison, Amar Saed, Meriam Sena, Yimin Xu'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
sys.path.insert(0, os.path.abspath('../../'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': '__init__',
    'inherited-members': True,
    'show-inheritance': True,
}


def skip_member(app, what, name, obj, skip, options):
    return skip


def setup(app):
    app.connect('autodoc-skip-member', skip_member)


# Add LaTeX options
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': '',
    'printindex': '',  # Exclude the index
    'makeindex': '',   # Disable index generation
}
# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ['_templates']
exclude_patterns = []

# language = 'python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
# Grouping the document tree into LaTeX files
latex_documents = [
    ('index', 'UltrasoundSim.tex', 'Ultrasound Imaging Simulator Documentation',
     'Parham Baghbanbashi, Luca Hutchison, \\\\  Amar Saed, Meriam Sena, Yimin Xu', 'manual'),
]
