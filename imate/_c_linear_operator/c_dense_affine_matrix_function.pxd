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
from .c_affine_matrix_function cimport cAffineMatrixFunction


# =======
# Externs
# =======

cdef extern from "c_dense_affine_matrix_function.h":

    cdef cppclass cDenseAffineMatrixFunction[DataType](cAffineMatrixFunction):

        cDenseAffineMatrixFunction(
                const DataType* A_,
                const FlagType A_is_row_major_,
                const LongIndexType num_rows_,
                const LongIndexType num_colums_) except +

        cDenseAffineMatrixFunction(
                const DataType* A_,
                const FlagType A_is_row_major_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_,
                const DataType* B_,
                const FlagType B_is_row_major_) except +
