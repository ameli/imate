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

from .._definitions.types cimport IndexType, LongIndexType
from .random_number_generator cimport RandomNumberGenerator


# =======
# Externs
# =======

cdef extern from "random_array_generator.h":

    cdef cppclass RandomArrayGenerator[DataType]:

        # generate random array
        @staticmethod
        void generate_random_array(
                RandomNumberGenerator& random_number_generator,
                DataType* array,
                const LongIndexType array_size,
                const IndexType num_threads) nogil
