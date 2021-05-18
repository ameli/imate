# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


"""
Note: module names in __init__.pxd should be absolute import (not relative
import).

* In Cython init files should do absolute import, like:

    ::

        from package.subpackage.subsubpakcage.module_name import function

* In python init files can do relative import, like:

    ::

        from .subsubpackage import function
"""

from imate._c_trace_estimator.py_c_trace_estimator cimport pyc_trace_estimator

__all__ = ['pyc_trace_estimator']
