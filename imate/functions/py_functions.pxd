# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.

# =======
# Imports
# =======

from .functions cimport Function


# ============
# Declarations
# ============

cdef class pyFunction(object):

    # Member data
    cdef Function* matrix_function

    # Member functions
    cdef void set_function(self, Function* matrix_function_) except *
    cdef Function* get_function(self) nogil
