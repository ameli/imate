"""
Reference:

    This script was modified from scipy's optimization documentation, namely,
    the scipy's source code in `/scipy/doc/source/scipyoptdoc.py` file.

What this script does:

    This script creates a new custom sphinx domain. In sphinx, things like

    .. py:funtion:: some_function_name
    .. py:module:: some_module_name
    .. py:class:: some_class_name

    and so on are called domain. This script creates a domain called:

    .. custom-domaon::function <interface_function_name>
       :impl: <implementation_function_name>
       :method: <method_name>

Why this custom domain is needed:

    Suppose you have the function `logdet` with the signature

        logdet(A, p, method=method, **options)

    Depending on the argument 'method', the above function accepts different
    **options arguments. For example:

        logdet(A, p, method='cholesky', colmod=False)
        logdet(A, p, method='hutchinson', num_samples=20)
        logdet(A, p, method='slq', lanczos_degree=20)

    Also, the interface function, `logdet`, dispatches the computation to
    different implementation functions. Suppose logdet and the above functions
    are implemented respectively by these files:

    The interface function:
        imate.logdet.logdet(A, ...)

    The implementation functions:
        imate.logdet._cholesky_method.cholesky_method(A, ...)
        imate.logdet._hutchinson_method.hutchinson_method(A, ...)
        imate.logdet._slq_method.slq_method(A, ...)

    We include the docstring of interface function by

        .. autosummary::
            :toctree: generated
            :caption: Functions
            :recursive:
            :template: autosummary/member.rst

            imate.logdet

    Now, we also want to create a docstring for each of the implementation
    functions. If we do just as like in the above, like by adding

        .. autosummary::
            :toctree: generated
            :caption: Functions
            :recursive:
            :template: autosummary/member.rst

            imate.logdet
            imate.logdet._slq_method.slq_method

    then it creates a docstring with the signature

        imate.logdet._slq_method.slq_method(A, p, method='slq', lanczos_degr..)
        ... content of _slq_method

    but what we really want is

        imate.logdet(A, p, method='slq', lanczos_degr..)
        ... content of _slq_method

    That is, we want these three things together:

    (1) take the NAME of function from interface file
    (2) keep the SIGNATURE from the implementation file
    (3) keep the CONTENT from implementation file

    More details in the above three:

    (1) We want the NAME of the function in the docstring to be the name of
        the interface function, not the implementation function. Because the
        name of the implementation function is

            imate.logdet._slq_method_slq_method

        We need to make a custom domain that uses this name instead:

            imate.logdet

    (2) We want to take the signature of the implementation function, not the
        interface function. Namely, we want

        imate.logdet(A, p, method='slq', lanczos_degree=20)

    (3) We want to include the docstring content of the implementation, not
        the interface file.

How the custom domain solves this issue:

    By using .. autosummary::, .. function::, etc, we cannot do these. However,
    with this domain we can do this using

    .. custom-domain:function:: imate.logdet        <= interface function
       :impl: imate.logdet._slq_method.slq_method   <= implementation function
       :method: slq                                 <= value of 'method' arg
"""


import os, sys

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

# ------------

import re, pydoc
import sphinx
import inspect
import collections
import textwrap
import warnings

if sphinx.__version__ < '1.0.1':
    raise RuntimeError("Sphinx 1.0.1 or newer is required")

from numpydoc.numpydoc import mangle_docstrings
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList
from sphinx.domains.python import PythonDomain
from scipy._lib._util import getfullargspec_no_self


# =====
# setup
# =====

def setup(app):
    app.add_domain(CustomInterfaceDomain)
    return {'parallel_read_safe': True}


# ===================
# option required str
# ===================

def _option_required_str(x):
    if not x:
        raise ValueError("value is required")
    return str(x)


# =============
# import object
# =============

def _import_object(name):
    parts = name.split('.')
    module_name = '.'.join(parts[:-1])
    __import__(module_name)
    obj = getattr(sys.modules[module_name], parts[-1])
    return obj


# =======================
# Custom Interface Domain
# =======================

class CustomInterfaceDomain(PythonDomain):

    # This is the name of our custom domain that we will use in the doc.
    name = 'custom-domain'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.directives = dict(self.directives)
        self.directives['function'] = wrap_mangling_directive(
                self.directives['function'])


# =======================
# wrap mangling directive
# =======================

def wrap_mangling_directive(base_directive):

    class directive(base_directive):

        def run(self):
            env = self.state.document.settings.env

            # Interface function (we only use its name, but not its signature)
            iface_name = self.arguments[0].strip()
            iface_obj = _import_object(iface_name)
            iface_args, iface_varargs, iface_keywords, iface_defaults = \
                    getfullargspec_no_self(iface_obj)[:4]

            # Implementation function (we use its signature, but not its name)
            impl_name = self.options['impl']
            impl_obj = _import_object(impl_name)
            impl_args, impl_varargs, impl_keywords, impl_defaults = \
                    getfullargspec_no_self(impl_obj)[:4]

            # Insert `method=<method-name>` to the implementation signature
            method_name = self.options['method']

            # Insert 'method' to impl_args
            num_iface_args = len(iface_args)
            impl_args = list(impl_args)
            impl_args.insert(num_iface_args-1, 'method')
            impl_args = tuple(impl_args)

            # Insert method name to impl_defaults
            num_iface_defaults = len(iface_defaults)
            impl_defaults = list(impl_defaults)
            impl_defaults.insert(num_iface_defaults-1, method_name)
            impl_defaults = tuple(impl_defaults)

            # Create custom signature based on impl function, not interface
            # function.
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('ignore')
                signature = inspect.formatargspec(
                    impl_args, impl_varargs, impl_keywords, impl_defaults)

            # Custom signature consists of the name of interface function plus
            # the signature of the implementation function.
            self.options['noindex'] = True
            self.arguments[0] = iface_name + signature
            lines = textwrap.dedent(pydoc.getdoc(impl_obj)).splitlines()

            # Add `seelaso` directive to the content
            see_also = \
                """
                .. seealso::

                    This page describes only the `%s` method. For other
                    methods, see :func:`%s`.
                """ % (method_name, iface_name)

            # Find where to add see_also. We add it right after the first
            # paragraph. We exclude the first item in the list.
            insert_index = 0
            for i in range(1, len(lines)):
                if lines[i] == '':
                    insert_index = i
                    break

            lines.insert(insert_index, see_also)

            # Generate doc string with the custom signature
            mangle_docstrings(env.app, 'function', impl_name,
                              None, None, lines)
            self.content = ViewList(lines, self.content.parent)

            return base_directive.run(self)

        # ------

        option_spec = dict(base_directive.option_spec)
        option_spec['impl'] = _option_required_str
        option_spec['method'] = _option_required_str

    return directive
