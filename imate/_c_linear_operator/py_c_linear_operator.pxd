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

from .._definitions.types cimport IndexType, LongIndexType
from .c_linear_operator cimport cLinearOperator


# ===================
# pyc Linear Operator
# ===================

cdef class pycLinearOperator(object):

    # Attributes
    cdef cLinearOperator[float]* Aop_fp32
    cdef cLinearOperator[double]* Aop_fp64
    cdef cLinearOperator[long double]* Aop_fp128
    cdef IndexType num_parameters
    cdef data_type_name
    cdef long_index_type_name
    cdef parameters

    # Cython methods
    cdef LongIndexType get_num_rows(self) except *
    cdef LongIndexType get_num_columns(self) except *
    cdef cLinearOperator[float]* get_linear_operator_fp32(self) except *
    cdef cLinearOperator[double]* get_linear_operator_fp64(self) except *
    cdef cLinearOperator[long double]* get_linear_operator_fp128(self) except *
    cpdef void set_symmetry(self, symmetric) except *
    cpdef void set_parameters(self, parameters) except *
    cpdef void dot(self, vector, product) except *
    cpdef void transpose_dot(self, vector, product) except *
