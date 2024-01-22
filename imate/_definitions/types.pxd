# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# ========
# Includes
# ========

include "definitions.pxi"

# =====
# Types
# =====

"""
Use ``LongIndexType`` type for long indices where parallelization could be
important. This could include, for instance, indices of long columns of
matrices, but not short rows.

This type is intended to be set as ``unsigned long``. However, because the
indices of ``scipy.sparse`` matrices are stored as ``int`` (and not ``long``),
here, a fused type is used to accommodate both ``int`` and ``long``.

The type of indices of sparse arrays can be cast, for instance by:

  .. code-block:: python

    # In this file:
    ctypedef unsigned long IndexType

    # In linear_operator.pyx:LinearOperator:__cinit__()
    # Add .astype('uint64') to these variables:
    self.A_indices = A.indices.astype('uint64')
    self.B_indices = A.indices.astype('uint64')
    self.A_index_pointer = A.indptr.astype('uint64')
    self.B_index_pointer = B.indptr.astype('uint64')

In the above, ``uint64`` is equivalent to ``unsigned long``. Note, this will
*copy* the data, since scipy's sparse indices should be casted from ``uint32``.
"""

# ===========
# Fused types (templates)
# ===========

ctypedef fused DataType:
    float
    double
    long double

ctypedef fused ConstDataType:
    const float
    const double
    const long double

ctypedef fused MemoryViewDataType:
    float[:]
    double[:]
    long double[:]

ctypedef fused MemoryView2DCDataType:
    float[:, ::1]
    double[:, ::1]
    long double[:, ::1]

ctypedef fused MemoryView2DFDataType:
    float[::1, :]
    double[::1, :]
    long double[::1, :]


# ============
# Static types (non-templates)
# ============

# The LongIndexType is defined in definitions.h and is used for the C++ source
# codes. This type can be 4 byte int or 8 byte long int, depending on macros
# that is defined at compile time. To expose the LongIndexType to pyx files as
# well, we extern the definitions.h. But this also requires to re-define this
# type with ctypedef and with yet another type (here, int), despite it might
# be defined as long int in definitions.h. Note that the int type here is
# arbitrary and will be ignored by the cython compiler. Read more about this at
# https://cython.readthedocs.io/en/latest/src/userguide/external_C_code.html
# in the typedef section of that page.

cdef extern from "./definitions.h":
    # Note: here LongIndexType is defined as "int" (which is 4 byte). This is
    # just a placeholder and is ignored by cython. Rather, the type defined in
    # definition.h is always used, which could be int or long int.
    ctypedef int LongIndexType

# Long index types is used for data indices, such a matrix and vectors indices
IF MEMORY_VIEW_LONG_INT:
    IF MEMORY_VIEW_UNSIGNED_LONG_INT:
        ctypedef unsigned long[:] MemoryViewLongIndexType
    ELSE:
        ctypedef long[:] MemoryViewLongIndexType
ELSE:
    IF MEMORY_VIEW_UNSIGNED_LONG_INT:
        ctypedef unsigned int[:] MemoryViewLongIndexType
    ELSE:
        ctypedef int[:] MemoryViewLongIndexType

# Used for indices of small matrices, or small size iterators
ctypedef int IndexType
ctypedef int[:] MemoryViewIndexType

# Used for both flags and integers used as signals, including negative integers
ctypedef int FlagType
ctypedef int[:] MemoryViewFlagType


# ==============
# Function Types
# ==============

ctypedef double (*kernel_type)(                                    # noqa: E211
        const double x,
        const double kernel_param) nogil
