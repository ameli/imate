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

from .._definitions.types cimport IndexType, LongIndexType, FlagType
from .c_linear_operator_base cimport cLinearOperatorBase


# =======
# Externs
# =======

cdef extern from "c_linear_operator.h":

    cdef cppclass cLinearOperator[DataType](cLinearOperatorBase):

        cLinearOperator() except +

        void set_parameters(DataType* parameters_) noexcept nogil

        void dot(
                const DataType* vector,
                DataType* product) noexcept nogil

        void transpose_dot(
                const DataType* vector,
                DataType* product) noexcept nogil
