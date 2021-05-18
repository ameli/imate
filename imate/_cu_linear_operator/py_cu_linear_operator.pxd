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
    cdef cuLinearOperator[long double]* Aop_long_double
    cdef char* data_type_name
    cdef IndexType num_parameters
    cdef dict device_properties_dict

    # Cython methods
    cdef char* get_data_type_name(self)
    cdef cuLinearOperator[float]* get_linear_operator_float(self)
    cdef cuLinearOperator[double]* get_linear_operator_double(self)
