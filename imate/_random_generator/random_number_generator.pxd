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

from libc.stdint cimport uint64_t


# =======
# Externs
# =======

cdef extern from "random_number_generator.h":

    cdef cppclass RandomNumberGenerator:

        RandomNumberGenerator() except +
        RandomNumberGenerator(int num_threads_) except +
        uint64_t next(int thread_id) nogil
