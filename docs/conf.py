# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'TraceInv'
copyright = '2020, Siavash Ameli'
author = 'Siavash Ameli'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx_math_dollar','sphinx.ext.mathjax',
    'sphinx.ext.graphviz','sphinx.ext.inheritance_diagram',
    'sphinx.ext.viewcode',
    'sphinx_toggleprompt',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary'
]

# autosummary
autosummary_generate = True

mathjax_config = {
    'tex2jax': {
        'inlineMath': [ ["\\(","\\)"] ],
        'displayMath': [["\\[","\\]"] ],
    },
}

# LaTeX
# 'sphinx.ext.imgmath',
# imgmath_image_format = 'svg'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Good themes
html_theme = 'sphinx_rtd_theme'
# html_theme = 'nature'               # <-- I used this one
# html_theme = 'bizstyle'
# html_theme = 'haiku'
# html_theme = 'classic'

# Not good thmes
# html_theme = 'sphinxdoc'
# html_theme = 'alabaster'
# html_theme = 'pyramid'
# html_theme = 'agogo'
# html_theme = 'traditional'
# html_theme = 'scrolls'

# To use sphinx3, download https://github.com/sphinx-doc/sphinx/tree/master/doc/_themes/sphinx13
# and put it in /docs/_themes
# html_theme = 'sphinx13'
# html_theme_path = ['_themes']

# import sphinx_readable_theme
# html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
# html_theme = 'readable'

# import sphinx_nameko_theme
# html_theme_path = [sphinx_nameko_theme.get_html_theme_path()]
# html_theme = 'nameko'

# import sphinx_bootstrap_theme
# html_theme = 'bootstrap'
# html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# extensions.append('sphinxjp.themes.basicstrap')
# html_theme = 'basicstrap'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# # Add css
# html_css_files = [
#     'custom.css',
# ]

def setup (app):
    app.add_stylesheet('css/custom.css')   # relative to /docs/_static/
