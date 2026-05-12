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
from .cu_matrix cimport cuMatrix


# =======
# Externs
# =======

cdef extern from "cu_csc_matrix.h":

    cdef cppclass cuCSCMatrix[DataType](cuMatrix):

        cuCSCMatrix() except +

        cuCSCMatrix(
                const DataType* A_data_,
                const LongIndexType* A_indices_,
                const LongIndexType* A_index_pointer_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_,
                const FlagType A_is_symmetric_,
                const int num_gpu_devices_) except +
