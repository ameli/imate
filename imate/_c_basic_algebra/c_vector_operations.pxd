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

from .._definitions.types cimport IndexType, LongIndexType, FlagType


# =======
# Externs
# =======

# Source code
cdef extern from "c_vector_operations.cpp":
    pass

# header
cdef extern from "c_vector_operations.h":

    cdef cppclass cVectorOperations[DataType]:

        @staticmethod
        void copy_vector(
                const DataType* input_vector,
                const LongIndexType vector_size,
                DataType* output_vector) nogil

        @staticmethod
        void copy_scaled_vector(
                const DataType* input_vector,
                const LongIndexType vector_size,
                const DataType scale,
                DataType* output_vector) nogil

        @staticmethod
        void subtract_scaled_vector(
                const DataType* input_vector,
                const LongIndexType vector_size,
                const DataType scale,
                DataType* output_vector) nogil

        @staticmethod
        DataType inner_product(
                const DataType* vector1,
                const DataType* vector2,
                const LongIndexType vector_size) nogil

        @staticmethod
        DataType euclidean_norm(
                const DataType* vector,
                const LongIndexType vector_size) nogil

        @staticmethod
        DataType normalize_vector_in_place(
                DataType* vector,
                const LongIndexType vector_size) nogil

        @staticmethod
        DataType normalize_vector_and_copy(
                const DataType* vector,
                const LongIndexType vector_size,
                DataType* output_vector) nogil
