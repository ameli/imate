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
copyright = '2020, Siavash Ameli'
author = 'Siavash Ameli'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx_math_dollar', 'sphinx.ext.mathjax',
    'sphinx.ext.graphviz', 'sphinx.ext.inheritance_diagram',
    'sphinx.ext.viewcode',
    'sphinx_toggleprompt',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx_automodapi.automodapi',
]

# autosummary
autosummary_generate = True

# automodapi
numpydoc_show_class_members = False

mathjax_config = {
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
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Good themes
html_theme = 'sphinx_rtd_theme'
# html_theme = 'pydata_sphinx_theme'
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
# html_css_files = [
#     'custom.css',
# ]


def setup(app):
    """
    This function is used to employ a css file to the themes.
    Note: paths are relative to /docs/_static
    """

    app.add_css_file('css/custom.css')
    # app.add_css_file('css/custom-anaconda-doc.css')
