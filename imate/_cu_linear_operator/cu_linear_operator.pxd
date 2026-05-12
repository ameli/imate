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
from .._c_linear_operator.c_linear_operator_base cimport cLinearOperatorBase


# =======
# Externs
# =======

cdef extern from "cu_linear_operator.h":

    cdef cppclass cuLinearOperator[DataType](cLinearOperatorBase):

        cuLinearOperator() except +

        cuLinearOperator(const int num_gpu_devices) except +
        
        void set_parameters(DataType* parameters_) noexcept nogil

        void dot(
                const DataType* vector,
                DataType* product) noexcept nogil

        void transpose_dot(
                const DataType* vector,
                DataType* product) noexcept nogil
