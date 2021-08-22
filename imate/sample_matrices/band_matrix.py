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

import numpy
from scipy.sparse import coo_matrix

__all__ = ['band_matrix', 'band_matrix_trace', 'band_matrix_traceinv',
           'band_matrix_logdet']


# ===========
# band matrix
# ===========

def band_matrix(
        a,
        b,
        size=20,
        gram=True,
        format='csr',
        dtype=r'float64'):
    """
    """

    _check_arguments(a, b, size, gram, format, dtype)

    data = numpy.empty((2*size-1, ), dtype=dtype)
    row = numpy.empty((2*size-1, ), dtype=int)
    col = numpy.empty((2*size-1, ), dtype=int)

    # Fill diagonals
    row[:size] = numpy.arange(size)
    col[:size] = numpy.arange(size)
    data[:size] = a

    # Fill off-diagonal
    row[size:] = numpy.arange(size-1)
    col[size:] = numpy.arange(1, size)
    data[size:] = b

    A = coo_matrix((data, (row, col)), shape=(size, size))

    if gram:
        A = A.T @ A

    if format == 'csc':
        A = A.tocsc()
    elif format == 'csr':
        A = A.tocsr()

    return A


# ===============
# check arguments
# ===============

def _check_arguments(
        a,
        b,
        size,
        gram,
        format,
        dtype):
    """
    Checks the type and values of the input arguments.
    """

    # Check a
    if a is None:
        raise ValueError('"a" cannot be None.')
    elif not isinstance(a, (int, numpy.integer, float)):
        raise TypeError('"a" should be integer or float.')

    # Check b
    if b is None:
        raise ValueError('"b" cannot be None.')
    elif not isinstance(b, (int, numpy.integer, float)):
        raise TypeError('"b" should be integer or float.')

    # Check size
    if size is None:
        raise TypeError('"size" cannot be None.')
    elif not numpy.isscalar(size):
        raise TypeError('"size" should be a scalar value.')
    elif not isinstance(size, (int, numpy.integer)):
        TypeError('"size" should be an integer.')
    elif size < 1:
        raise ValueError('"size" should be a positive integer.')

    # Check gram
    if gram is None:
        raise TypeError('"gram" cannot be None.')
    elif not numpy.isscalar(gram):
        raise TypeError('"gram" should be a scalar value.')
    elif not isinstance(gram, bool):
        raise TypeError('"gram" should be boolean.')

    # Check format
    if format is not None:
        if not isinstance(format, str):
            raise TypeError('"format" should be string.')
        elif format not in ['csr', 'csc']:
            raise ValueError('"format" should be either "csr" or "csc".')

    # Check dtype
    if dtype is None:
        raise TypeError('"dtype" cannot be None.')
    elif not numpy.isscalar(dtype):
        raise TypeError('"dtype" should be a scalar value.')
    elif not isinstance(dtype, str):
        raise TypeError('"dtype" should be a string')
    elif dtype not in [r'float32', r'float64', r'float128']:
        raise TypeError('"dtype" should be either "float32", "float64", or ' +
                        '"float128".')


# =================
# band matrix trace
# =================

def band_matrix_trace(a, b, size, gram=True):
    """
    Computes the trace of band matrix based on known formula.
    """

    if gram:
        trace_ = a**2 + (size-1) * (a**2 + b**2)

    else:
        trace_ = size * a

    return trace_


# ====================
# band matrix traceinv
# ====================

def band_matrix_traceinv(a, b, size, gram=True):
    """
    Computes the trace of inverse band matrix based on known formula.
    """

    if gram:
        if a == b:
            traceinv_ = size * (size+1) / (2.0 * a**2)
        else:
            q = b / a
            if size < 200:
                traceinv_ = (1.0 / (a**2 - b**2)) * \
                    (size - (q**2) * ((q**(2.0*size) - 1.0) / (q**2 - 1.0)))
            else:
                # Using asymptotic approximation of large sums
                traceinv_ = (1.0 / (a**2 - b**2)) * \
                            (size - ((q**2) / (1.0 - q**2)))
    else:
        traceinv_ = size / a

    return traceinv_


# ==================
# band matrix logdet
# ==================

def band_matrix_logdet(a, b, size, gram=True):
    """
    Computes the log-determinant of band matrix based on known formula.
    """

    logdet_ = size * numpy.log(a)
    if gram:
        logdet_ = 2.0 * logdet_

    return logdet_
