# SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.

# =======
# Imports
# =======

import os
import sys
from datetime import date

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# The following is treatment exclusively for CYTHON. Since I build cython
# into '/build' directory, the lib files (*.so) are generated in the subfolder
# '/build/lib.linux-x86_64.3.8/'. To properly build sphinx, this directory
# should be included. Because this name of this subdirectory depends on the
# Linux platform, the architecture and the python version, in the following, we
# search for (glob) all subdirectories of '/build' and find which
# subdirectories contain '*.so' files. We then include all of those
# subdirectories to the path.

# Here as assumed that the '*.so' files are built inside the build directory.
# To do so,
# 1. Make sure cython package is built without '--inplace'. That is:
#    'python setup.py build_ext'.
# 2. Make sure in 'setup.cfg', the '[build_ext]' section does not have
#    'inplace=1' entry (if yes, comment it).

# If the build is make with '--inplace', then the '*.so' files are written
# inside the source code where '*.pyx' files are. In this case, you do not need
# to include the subdirectories of '/build' on the path.

# The RecursiveGolb.py should be located in '/docs'.
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))
import recursive_glob  # this must be after including ./ path      # noqa: E402

# Build (assuming we build cython WITHOUT '--inplace', that is:
# 'python setup.py build_ext' only.
build_directory = os.path.join('..', 'build')

# Regex for pattern of lib files. Note: this is OS dependant. macos: *.dylib.
# Windows: *.dll
lib_file_patterns = ['*.so', '*.dylib', '*.dll']

# Find list of subdirectories of build directory that have files with pattern
build_subdirectories = recursive_glob.recursive_glob(
        build_directory,
        lib_file_patterns)

# Append the subdirectories to the path
for build_subdirectory in build_subdirectories:

    # Note: the subdirectory is *relative* to the BuildDirectory.
    path = os.path.join(build_directory, build_subdirectory)
    sys.path.insert(0, os.path.abspath(path))
    print(os.path.abspath(path))

# -- Project information -----------------------------------------------------

project = 'imate'
author = 'Siavash Ameli'
copyright = f'{date.today().year}, ' + author

# -- Sphinx Settings ----------------------------------------------------------

# Check links and references
nitpicky = True

# Why setting "root_doc": by using toctree in index.rst, two things are added
# to index.html main page: (1) a toc in that location of page, (2) toc in the
# sidebar menu. If we add :hidden: option to toctree, it removes toc from
# both page and sidebar menu. There is no way we can have only one of these,
# for instance, toc only in the page, but not in the menu. A solution to
# this is as follows:
#   1. Set "root_doc= 'content'". Then add those toc that should go into the
#      menu in the content.rst file.
#   2. Add those toc that should go into the page in index.rst file.
# This way, we can control which toc appears where.
#
# A problem: by setting "root_doc='content'", the sidebar logo links to
# contents.html page, not the main page. There is a logo_url variable but it
# does not seem to do anything. To fix this, I added a javascript (see in
# /docs/source/_static/js/custom-pydata.js) which overwrites
# <a href"path/contents.html"> to <a href="path/index.html>".
root_doc = "contents"

# Common definitions for the whole pages
rst_epilog = '''
.. role:: synco
   :class: synco

.. |project| replace:: :synco:`imate`
'''

# Figure, Tables, etc numbering
# numfig = True
# numfig_format = {
#     'figure': 'Figure %s',
#     'table': 'Table %s'
# }

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'custom_domain',
    'sphinx.ext.autodoc',
    # 'sphinx_math_dollar',
    'sphinx.ext.mathjax',
    'sphinx.ext.graphviz', 'sphinx.ext.inheritance_diagram',
    # 'sphinx.ext.viewcode',
    'sphinx_toggleprompt',
    # 'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    # 'sphinx_automodapi.automodapi',
    # 'sphinxcontrib.napoleon',               # either use napoleon or numpydoc
    'numpydoc',                               # either use napoleon or numpydoc
    'sphinx_design',
    # 'sphinx_multitoc_numbering',
    'sphinx-prompt',
    'sphinx_copybutton',
    'nbsphinx',
    'sphinx_gallery.load_style',
    "sphinxext.opengraph",
]

# Copy button settings
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r'>>> |\.\.\. '

# Automatically generate autosummary after each build
autosummary_generate = True
autosummary_imported_members = True

# Remove the module names from the signature
# add_module_names = False

# automodapi
numpydoc_show_class_members = False

# Added after including sphinx_math_dollar. The following prevents msthjax to
# parse $ and $$.
mathjax3_config = {
    'tex2jax': {
        'inlineMath': [["\\(", "\\)"]],
        'displayMath': [["\\[", "\\]"]],
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
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Good themes
# html_theme = 'sphinx_rtd_theme'
html_theme = 'pydata_sphinx_theme'
# html_theme = 'sphinx_book_theme'
# html_theme = 'nature'
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

# Options for theme
html_theme_options = {
    "github_url": "https://github.com/ameli/imate",
    "navbar_end": [
        "theme-switcher",
        "search-field.html",
        "navbar-icon-links.html"
    ],
    "page_sidebar_items": ["page-toc", "edit-this-page"],
    # "header_links_before_dropdown": 4,
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/imate/",
            "icon": "fab fa-python",
            "type": "fontawesome",
        },
        {
            "name": "Anaconda Cloud",
            "url": "https://anaconda.org/s-ameli/imate",
            "icon": "fa fa-circle-notch",
            "type": "fontawesome",
        },
        {
            "name": "Docker Hub",
            "url": "https://hub.docker.com/r/sameli/imate",
            "icon": "fab fa-docker",
            "type": "fontawesome",
        },
        {
            "name": "Lanuch Jupyter on Binder",
            "url": "https://mybinder.org/v2/gh/ameli/imate/HEAD?filepath=" + \
                   "notebooks%2FInterpolateTraceOfInverse.ipynb",
            "icon": "fa fa-chart-line",
            "type": "fontawesome",
        },
    ],
    "pygment_light_style": "tango",
    "pygment_dark_style": "native",
    "logo": {
        "image_light": "images/icons/logo-imate-light.png",
        "image_dark": "images/icons/logo-imate-dark.png",
    },
}

html_context = {
    "default_mode": "auto",
    "github_url": "https://github.com",
    "github_user": "ameli",
    "github_repo": "imate",
    "github_version": "main",
    "doc_path": "docs/source",
}

html_sidebars = {
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"]
}

# Using Font Awesome icons
# html_css_files = [
#     "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"
# ]

html_title = f"{project} Manual"
html_last_updated_fmt = '%b %d, %Y'
# html_show_sourcelink = False

# To use sphinx3, download at
# https://github.com/sphinx-doc/sphinx/tree/master/doc/_themes/sphinx13
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
# html_css_files = ["css/custom.css"]
# html_css_files = ['css/custom-anaconda-doc.css']

html_js_files = ["js/custom-pydata.css"]
# html_logo = '_static/images/icons/logo-imate-light.png'
html_favicon = '_static/images/icons/favicon.ico'

# Open Graph cards for sharing the documentation on social media
ogp_site_url = 'https://ameli.github.io/imate'
ogp_image = 'https://raw.githubusercontent.com/ameli/imate/main/docs/' + \
            'source/_static/images/icons/logo-imate-light.svg'
ogp_site_name = 'RestoreIO'
ogp_description_length = 300
ogp_type = "website"
ogp_enable_meta_description = True
ogp_custom_meta_tags = [
    '<meta property="og:title" content="RestoreIO">',
    '<meta property="og:description" content="imate, short for Implicit ' +
    'Matrix Trace Estimator, is a modular and high-performance C++/CUDA ' +
    'library distributed as a Python package that provides scalable ' +
    'randomized algorithms for the computationally expensive matrix ' +
    'functions in machine learning.">',
]


# =====
# setup
# =====

def setup(app):
    """
    This function is used to employ a css file to the themes.
    Note: paths are relative to /docs/source/_static
    """

    app.add_css_file('css/custom-pydata.css')
    app.add_js_file('js/custom-pydata.js')
    # app.add_css_file('css/custom.css')
    # app.add_css_file('css/custom-anaconda-doc.css')
