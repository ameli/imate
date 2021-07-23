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

from .._definitions.types cimport DataType, ConstDataType, IndexType, \
        LongIndexType, FlagType
from .c_linear_operator cimport cLinearOperator


# ===================
# pyc Linear Operator
# ===================

cdef class pycLinearOperator(object):

    # Attributes
    cdef cLinearOperator[float]* Aop_float
    cdef cLinearOperator[double]* Aop_double
    cdef cLinearOperator[long double]* Aop_long_double
    cdef char* data_type_name
    cdef char* long_index_type_name
    cdef IndexType num_parameters
    cdef parameters

    # Cython methods
    cdef LongIndexType get_num_rows(self) except *
    cdef LongIndexType get_num_columns(self) except *
    cdef cLinearOperator[float]* get_linear_operator_float(self) except *
    cdef cLinearOperator[double]* get_linear_operator_double(self) except *
    cdef cLinearOperator[long double]* get_linear_operator_long_double(
            self) except *
    cpdef void dot(self, vector, product) except *
    cpdef void transpose_dot(self, vector, product) except *
