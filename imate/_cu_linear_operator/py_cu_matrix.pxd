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

from .py_cu_linear_operator cimport pycuLinearOperator
from .._definitions.types cimport FlagType


# ==========
# pycuMatrix
# ==========

cdef class pycuMatrix(pycuLinearOperator):
    cdef FlagType A_is_symmetric
    cdef A_csr
    cdef A_indices_copy
    cdef A_index_pointer_copy
    cdef Py_buffer A_data_py_buffer
    cdef Py_buffer A_indices_py_buffer
    cdef Py_buffer A_index_pointer_py_buffer
