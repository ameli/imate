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

cdef extern from "c_csr_affine_matrix_function.h":

    cdef cppclass cCSRAffineMatrixFunction[DataType](cAffineMatrixFunction):

        cCSRAffineMatrixFunction(
                const DataType* A_data_,
                const LongIndexType* A_indices_,
                const LongIndexType* A_index_pointer_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_) except +

        cCSRAffineMatrixFunction(
                const DataType* A_data_,
                const LongIndexType* A_indices_,
                const LongIndexType* A_index_pointer_,
                const LongIndexType num_rows_,
                const LongIndexType num_colums_,
                const DataType* B_data_,
                const LongIndexType* B_indices_,
                const LongIndexType* B_index_pointer_) except +
