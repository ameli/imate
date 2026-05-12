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
from .cu_linear_operator cimport cuLinearOperator


# =======
# Externs
# =======

cdef extern from "cu_matrix.h":

    cdef cppclass cuMatrix[DataType](cuLinearOperator):

        cuMatrix() except +
        cuMatrix(const FlagType A_is_symmetric_) except +
