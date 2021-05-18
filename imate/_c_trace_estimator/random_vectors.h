/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_TRACE_ESTIMATOR_RANDOM_VECTORS_H_
#define _C_TRACE_ESTIMATOR_RANDOM_VECTORS_H_

// =======
// Headers
// =======

#include "../_definitions/types.h"  // IndexType, LongIndexType

// ==============
// Random Vectors
// ==============

/// \class RandomVectors
///
/// \brief A static class to generate random set of vectors. This class acts as
///        a templated namespace, where all member methods are *public* and
///        *static*.
///
/// \sa    Orthogonalization

template <typename DataType>
class RandomVectors
{
    public:

        // generate random column vectors
        static void generate_random_column_vectors(
                DataType* vectors,
                const LongIndexType vector_size,
                const IndexType num_vectors,
                const IndexType orthogonalize,
                const IndexType num_parallel_threads);

        // generate random row vectors
        static void generate_random_row_vectors(
                DataType** vectors,
                const LongIndexType vector_size,
                const IndexType num_vectors,
                const IndexType num_parallel_threads);
};

#endif  // _C_TRACE_ESTIMATOR_RANDOM_VECTORS_H_
