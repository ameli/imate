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

# ===========================
# pycu Affine Matrix Function
# ===========================

cdef class pycuAffineMatrixFunction(pycuLinearOperator):
    cdef A_csr
    cdef B_csr
    cdef A_indices_copy
    cdef A_index_pointer_copy
    cdef B_indices_copy
    cdef B_index_pointer_copy
