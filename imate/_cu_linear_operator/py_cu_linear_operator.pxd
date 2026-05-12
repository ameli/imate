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
from .cu_linear_operator cimport cuLinearOperator
from .._cu_definitions.cu_types cimport __nv_fp8_e5m2, __nv_fp8_e4m3, __half, \
        __nv_bfloat16


# ====================
# pycu Linear Operator
# ====================

cdef class pycuLinearOperator(object):

    # Attributes
    cdef cuLinearOperator[__nv_fp8_e5m2]* Aop_fp8_e5m2
    cdef cuLinearOperator[__nv_fp8_e4m3]* Aop_fp8_e4m3
    cdef cuLinearOperator[__half]* Aop_fp16
    cdef cuLinearOperator[__nv_bfloat16]* Aop_bf16
    cdef cuLinearOperator[float]* Aop_fp32
    cdef cuLinearOperator[double]* Aop_fp64
    cdef IndexType num_parameters
    cdef data_type_name
    cdef long_index_type_name
    cdef parameters
    cdef int num_gpu_devices
    cdef dict device_properties_dict

    # Cython methods
    cdef LongIndexType get_num_rows(self) except *
    cdef LongIndexType get_num_columns(self) except *
    cdef cuLinearOperator[__nv_fp8_e5m2]* get_linear_operator_fp8_e5m2(
            self) except *
    cdef cuLinearOperator[__nv_fp8_e4m3]* get_linear_operator_fp8_e4m3(
            self) except *
    cdef cuLinearOperator[__half]* get_linear_operator_fp16(self) except *
    cdef cuLinearOperator[__nv_bfloat16]* get_linear_operator_bf16(
            self) except *
    cdef cuLinearOperator[float]* get_linear_operator_fp32(self) except *
    cdef cuLinearOperator[double]* get_linear_operator_fp64(self) except *
    cpdef void set_symmetry(self, symmetric) except *
    cpdef void set_parameters(self, parameters) except *
    cpdef void dot(self, vector, product) except *
    cpdef void transpose_dot(self, vector, product) except *
