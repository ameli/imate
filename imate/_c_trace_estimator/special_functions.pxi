# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

from scipy.special.cython_special cimport erfinv                    # noqa E999


# =======
# erf inv
# =======

cdef public double erf_inv(double x) nogil:
    """
    Wrapper for cython's ``erfinv`` function. This wrapper is externed in C++
    code.
    """

    return erfinv(x)
