#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


"""
Why this script is needed:

    As part of the custom_domain.py script, the function
    "inspect.formatargspec()" from python's "inspect" package is needed.
    However, this function is deprecated since py3.11, and was replaced by
    "inspect.signature()". Unfortunately, with "inspect.signature()", we cannot
    customize the function signature, since it only accepts the function or
    class object as argument. To observe the issue, consider the new function
    which takes only one argument "obj" as

        import inspect
        signature = inspect.signature(obj)

    where, obj is a function or class object. On the other hand, the deprecated
    function "inspect.formatargspec()" accepts the COMPONENTS of the signature
    like:

        import inspect
        signature = inspect.formatargspec(args, varargs, keywords, defaults)

    where the args, varags, keywords and defaults can be customized before
    calling this function.

    The deprecated function is desired for our customization. To be able to use
    this function, I copied a part of the file "inspect.py" from here:

        https://github.com/python/cpython/blob/3.10/Lib/inspect.py

    which is the "inspect.py" implementation of CPython version 3.10. Note that
    in python 3.11, this function is removed.
"""


# =======
# Imports
# =======

import re
import types

__all__ = ['formatargspec']


# =================
# format annotation
# =================

def formatannotation(annotation, base_module=None):
    if getattr(annotation, '__module__', None) == 'typing':
        def repl(match):
            text = match.group()
            return text.removeprefix('typing.')
        return re.sub(r'[\w\.]+', repl, repr(annotation))
    if isinstance(annotation, types.GenericAlias):
        return str(annotation)
    if isinstance(annotation, type):
        if annotation.__module__ in ('builtins', base_module):
            return annotation.__qualname__
        return annotation.__module__+'.'+annotation.__qualname__
    return repr(annotation)


# ===============
# format arg spec
# ===============

def formatargspec(args, varargs=None, varkw=None, defaults=None,
                  kwonlyargs=(), kwonlydefaults={}, annotations={},
                  formatarg=str,
                  formatvarargs=lambda name: '*' + name,
                  formatvarkw=lambda name: '**' + name,
                  formatvalue=lambda value: '=' + repr(value),
                  formatreturns=lambda text: ' -> ' + text,
                  formatannotation=formatannotation):
    """Format an argument spec from the values returned by getfullargspec.

    The first seven arguments are (args, varargs, varkw, defaults,
    kwonlyargs, kwonlydefaults, annotations).  The other five arguments
    are the corresponding optional formatting functions that are called to
    turn names and values into strings.  The last argument is an optional
    function to format the sequence of arguments.

    Deprecated since Python 3.5: use the `signature` function and `Signature`
    objects.
    """

    from warnings import warn

    warn("`formatargspec` is deprecated since Python 3.5. Use `signature` and "
         "the `Signature` object directly",
         DeprecationWarning,
         stacklevel=2)

    def formatargandannotation(arg):
        result = formatarg(arg)
        if arg in annotations:
            result += ': ' + formatannotation(annotations[arg])
        return result
    specs = []
    if defaults:
        firstdefault = len(args) - len(defaults)
    for i, arg in enumerate(args):
        spec = formatargandannotation(arg)
        if defaults and i >= firstdefault:
            spec = spec + formatvalue(defaults[i - firstdefault])
        specs.append(spec)
    if varargs is not None:
        specs.append(formatvarargs(formatargandannotation(varargs)))
    else:
        if kwonlyargs:
            specs.append('*')
    if kwonlyargs:
        for kwonlyarg in kwonlyargs:
            spec = formatargandannotation(kwonlyarg)
            if kwonlydefaults and kwonlyarg in kwonlydefaults:
                spec += formatvalue(kwonlydefaults[kwonlyarg])
            specs.append(spec)
    if varkw is not None:
        specs.append(formatvarkw(formatargandannotation(varkw)))
    result = '(' + ', '.join(specs) + ')'
    if 'return' in annotations:
        result += formatreturns(formatannotation(annotations['return']))
    return result
