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
from .cu_affine_matrix_function cimport cuAffineMatrixFunction


# =======
# Externs
# =======

cdef extern from "cu_dense_affine_matrix_function.h":

    cdef cppclass cuDenseAffineMatrixFunction[DataType](
            cuAffineMatrixFunction):

        cuDenseAffineMatrixFunction(
                const DataType* A_,
                const FlagType A_is_row_major_,
                const LongIndexType num_rows_,
                const LongIndexType num_colums_,
                const int num_gpu_devices_) except +

        cuDenseAffineMatrixFunction(
                const DataType* A_,
                const FlagType A_is_row_major_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_,
                const DataType* B_,
                const FlagType B_is_row_major_,
                const int num_gpu_devices_) except +
