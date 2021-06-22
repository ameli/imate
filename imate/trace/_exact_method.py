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

import time
import numpy
import numpy.linalg
import scipy
import scipy.sparse


# ============
# exact method
# ============

def exact_method(A, exponent=1.0):
    """
    """
    # Checking input arguments
    check_arguments(A, exponent)

    init_wall_time = time.perf_counter()
    init_proc_time = time.process_time()

    if exponent == 0.0:
        trace = numpy.min(A.shape)

    elif exponent == 1:
        if scipy.sparse.issparse(A):
            trace = numpy.sum(A.diagonal())
        else:
            trace = numpy.trace(A)

    elif exponent == 2:
        if scipy.sparse.issparse(A):
            trace = numpy.sum(A.data**2)
        else:
            trace = numpy.linalg.norm(A, ord='fro')**2
    else:
        raise ValueError('For "exponent" other than "0", "1", and "2", use ' +
                         '"eigenvalue" or "slq" method.')

    wall_time = time.perf_counter() - init_wall_time
    proc_time = time.process_time() - init_proc_time

    info = {
        'cpu':
        {
            'wall_time': wall_time,
            'proc_time': proc_time,
        },
        'solver':
        {
            'method': 'exact'
        }
    }

    return trace, info


# ===============
# check arguments
# ===============

def check_arguments(A, exponent):
    """
    Checks the type and value of the parameters.
    """

    # Check A
    if (not isinstance(A, numpy.ndarray)) and (not scipy.sparse.issparse(A)):
        raise TypeError('Input matrix should be either a "numpy.ndarray" or ' +
                        'a "scipy.sparse" matrix.')
    elif A.shape[0] != A.shape[1]:
        raise ValueError('Input matrix should be a square matrix.')

    # Check exponent
    if exponent is None:
        raise TypeError('"exponent" cannot be None.')
    elif not numpy.isscalar(exponent):
        raise TypeError('"exponent" should be a scalar value.')
    elif not isinstance(exponent, int):
        TypeError('"exponent" cannot be an integer.')
