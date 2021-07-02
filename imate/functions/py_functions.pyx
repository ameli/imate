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
# py functions
# ============

cdef class pyFunction(object):
    """
    """

    def __cinit__(self):
        """
        """

        self.matrix_function = NULL

    cdef void set_function(self, Function* matrix_function_) except *:
        """
        """
        self.matrix_function = matrix_function_

    cdef Function* get_function(self) nogil:
        """
        """
        return self.matrix_function
