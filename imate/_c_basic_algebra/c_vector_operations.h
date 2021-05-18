/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_BASIC_ALGEBRA_C_VECTOR_OPERATIONS_H_
#define _C_BASIC_ALGEBRA_C_VECTOR_OPERATIONS_H_

// =======
// Headers
// =======

#include "../_definitions/types.h"  // LongIndexType


// =================
// Vector Operations
// =================

/// \class cVectorOperations
///
/// \brief A static class for vector operations, similar to level-1 operations
///        of the BLAS library. This class acts as a templated namespace, where
///        all member methods are *public* and *static*.
///
/// \sa    MatrixOperations

template <typename DataType>
class cVectorOperations
{
    public:

        // Copy Vector
        static void copy_vector(
                const DataType* input_vector,
                const LongIndexType vector_size,
                DataType* output_vector);

        // Copy Scaled Vector
        static void copy_scaled_vector(
                const DataType* input_vector,
                const LongIndexType vector_size,
                const DataType scale,
                DataType* output_vector);

        // Subtract Scaled Vector
        static void subtract_scaled_vector(
                const DataType* input_vector,
                const LongIndexType vector_size,
                const DataType scale,
                DataType* output_vector);

        // Inner Product
        static DataType inner_product(
                const DataType* vector1,
                const DataType* vector2,
                const LongIndexType vector_size);

        // Euclidean Norm
        static DataType euclidean_norm(
                const DataType* vector,
                const LongIndexType vector_size);

        // Normalize Vector In Place
        static DataType normalize_vector_in_place(
                DataType* vector,
                const LongIndexType vector_size);

        // Normalize Vector And Copy
        static DataType normalize_vector_and_copy(
                const DataType* vector,
                const LongIndexType vector_size,
                DataType* output_vector);
};

#endif  // _C_BASIC_ALGEBRA_C_VECTOR_OPERATIONS_H_
