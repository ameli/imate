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

from .random_number_generator cimport RandomNumberGenerator


# ==========================
# py Random Number Generator
# ==========================

cdef class pyRandomNumberGenerator(object):
    """
    A python wrapper for ``RandomNumberGenerator`` class.
    """

    # =====
    # cinit
    # =====

    def __init__(self, num_threads=1):
        """
        Initializes an instance of ``RandomNumberGenerator``.
        """

        self.random_number_generator = new RandomNumberGenerator(num_threads)

    # =======
    # dealloc
    # =======

    def __dealloc__(self):
        """
        Deallocates ``random_number_generator`` object.
        """

        del self.random_number_generator

    # ===========================
    # get random number generator
    # ===========================

    cdef RandomNumberGenerator* get_random_number_generator(self) nogil:
        """
        Returns a pointer to an instance of ``RandomNumberGenerator``.
        """

        return self.random_number_generator
