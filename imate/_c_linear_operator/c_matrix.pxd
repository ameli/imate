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

from .._definitions.types cimport LongIndexType
from .c_linear_operator cimport cLinearOperator


# =======
# Externs
# =======

cdef extern from "c_matrix.h":

    cdef cppclass cMatrix[DataType](cLinearOperator):

        cMatrix() except +
