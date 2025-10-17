# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Adjust this if your docs/ is elsewhere

project = 'HLA Solver Prototype'
copyright = '2025, Jay M. Appleton'
author = 'Jay M. Appleton'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

autosummary_generate = True

autodoc_default_options = {
    'members': True,
    'special-members': '__init__',
    'inherited-members': True,
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = ['**.ipynb', '**.ipynb_checkpoints', '_build', 'Thumbs.db', '.DS_Store']

html_theme = 'furo'
html_static_path = ['_static']

# Optional:
# html_logo = '_static/logo.png'
# html_favicon = '_static/favicon.ico'

# master_doc = 'index'  # only if not index.rst
