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


# =======
# Externs
# =======

cdef extern from "c_linear_operator.h":

    cdef cppclass cLinearOperator[DataType]:

        cLinearOperator() except +

        cLinearOperator(
                const LongIndexType num_rows_,
                const LongIndexType num_columns_) except +

        LongIndexType get_num_rows() nogil
        LongIndexType get_num_columns() nogil
        void set_parameters(DataType* parameters_) nogil
        IndexType get_num_parameters() nogil

        void dot(
                const DataType* vector,
                DataType* product) nogil

        void transpose_dot(
                const DataType* vector,
                DataType* product) nogil
