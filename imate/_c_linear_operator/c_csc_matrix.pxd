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
from .c_matrix cimport cMatrix


# =======
# Externs
# =======

cdef extern from "c_csc_matrix.h":

    cdef cppclass cCSCMatrix[DataType](cMatrix):

        cCSCMatrix() except +

        cCSCMatrix(
                const DataType* A_data_,
                const LongIndexType* A_indices_,
                const LongIndexType* A_index_pointer_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_) except +
