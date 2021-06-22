/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _RANDOM_GENERATOR_RANDOM_ARRAY_GENERATOR_H_
#define _RANDOM_GENERATOR_RANDOM_ARRAY_GENERATOR_H_


// =======
// Headers
// =======

#include "../_definitions/types.h"  // IndexType, LongIndexType
#include "./random_number_generator.h"  // RandomNumberGenerator


// ==============
// Random Vectors
// ==============

/// \class RandomArrayGenerator
///
/// \brief A static class to generate random set of vectors. This class acts as
///        a templated namespace, where all member methods are *public* and
///        *static*.
///
/// \sa    Orthogonalization

template <typename DataType>
class RandomArrayGenerator
{
    public:

        // generate random array
        static void generate_random_array(
                RandomNumberGenerator& random_number_generator,
                DataType* array,
                const LongIndexType array_size,
                const IndexType num_threads);
};


#endif  // _RANDOM_GENERATOR_RANDOM_ARRAY_GENERATOR_H_
