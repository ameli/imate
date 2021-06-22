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

    # Member data
    cdef RandomNumberGenerator* random_number_generator

    # Member methods
    cdef RandomNumberGenerator* get_random_number_generator(self) nogil
