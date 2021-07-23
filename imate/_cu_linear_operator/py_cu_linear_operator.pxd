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

from .._definitions.types cimport DataType, IndexType, LongIndexType, FlagType
from .cu_linear_operator cimport cuLinearOperator
from .._cuda_utilities cimport DeviceProperties


# ====================
# pycu Linear Operator
# ====================

cdef class pycuLinearOperator(object):

    # Attributes
    cdef cuLinearOperator[float]* Aop_float
    cdef cuLinearOperator[double]* Aop_double
    cdef char* data_type_name
    cdef char* long_index_type_name
    cdef IndexType num_parameters
    cdef int num_gpu_devices
    cdef dict device_properties_dict

    # Cython methods
    cdef LongIndexType get_num_rows(self) except *
    cdef LongIndexType get_num_columns(self) except *
    cdef char* get_data_type_name(self) except *
    cdef cuLinearOperator[float]* get_linear_operator_float(self) except *
    cdef cuLinearOperator[double]* get_linear_operator_double(self) except *
    cpdef void dot(self, vector, product) except *
    cpdef void transpose_dot(self, vector, product) except *
