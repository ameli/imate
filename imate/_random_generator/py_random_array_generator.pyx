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

from .._definitions.types cimport DataType, LongIndexType, IndexType
from .random_array_generator cimport RandomArrayGenerator
from .random_number_generator cimport RandomNumberGenerator
# from .py_random_number_generator cimport pyRandomNumberGenerator


# =====================
# generate random array
# =====================

cdef void py_generate_random_array(
        DataType* array,
        const LongIndexType array_size,
        const IndexType num_threads) nogil:
    """
    A python wrapper for ``RandomArrayGenerator.generate_random_array()``.

    :param array: A 1D array of size ``array_size``.
    :type array: c pointer

    :param array_size: The size of array
    :type array_size: int

    :param num_threads: Number of OpenMP parallel threads.
    :type num_threads: int
    """

    # Create a random number generator object
    cdef RandomNumberGenerator* random_number_generator = \
        new RandomNumberGenerator(num_threads)

    # Pass the random number generator to an array generator
    RandomArrayGenerator[DataType].generate_random_array(
            random_number_generator[0], array, array_size, num_threads)

    # Delete object
    del random_number_generator
