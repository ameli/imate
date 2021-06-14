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

from .._definitions.types cimport LongIndexType, FlagType
from .._c_linear_operator.c_dense_matrix cimport cDenseMatrix
from .cu_linear_operator cimport cuLinearOperator


# =======
# Externs
# =======

cdef extern from "cu_dense_matrix.h":

    cdef cppclass cuDenseMatrix[DataType](cDenseMatrix, cuLinearOperator):

        cuDenseMatrix() except +

        cuDenseMatrix(
                const DataType* A_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_,
                const FlagType A_is_row_major_,
                const int num_gpu_devices_) except +
