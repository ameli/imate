/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


// =======
// Headers
// =======

#include "./random_number_generator.h"
#include <cassert>  // assert
#include <cstdlib>  // NULL
#include "./xoshiro_256_star_star.h"  // Xoshiro256StarStar


// =============
// Constructor 1
// =============

/// \brief Initializes with one parallel thread and default seed.
///

RandomNumberGenerator::RandomNumberGenerator():
    num_threads(1)
{
    int64_t seed = -1;  // Negative indicates using processor time as seed
    this->initialize(seed);
}


// =============
// Constructor 2
// =============

/// \brief Initializes with a given number of parallel thread, but with the
///        default seed.
///
/// \param[in] num_threads_
///            Number of independent \c xoshiro_256_star_star objects to be
///            created for each parallel thread.

RandomNumberGenerator::RandomNumberGenerator(const int num_threads_):
    num_threads(num_threads_)
{
    int64_t seed = -1;  // Negative indicates using processor time as seed
    this->initialize(seed);
}


// =============
// Constructor 3
// =============

/// \brief Initializes with given number of parallel thread and seed.
///
/// \param[in] num_threads_
///            Number of independent \c xoshiro_256_star_star objects to be
///            created for each parallel thread.
/// \param[in] seed
///            Seed for pseudo-random number generation. The same seed value is
///            used for all threads. If seed is negative integer, the given
///            seed value is ignored, and the processor time is used istead.

RandomNumberGenerator::RandomNumberGenerator(
        const int num_threads_,
        const int64_t seed):
    num_threads(num_threads_)
{
    this->initialize(seed);
}


// ==========
// Destructor
// ==========

/// \brief Deallocates the array of \c xoshiro_256_star_star.
///

RandomNumberGenerator::~RandomNumberGenerator()
{
    if (this->xoshiro_256_star_star != NULL)
    {
        for (int thread_id=0; thread_id < this->num_threads; ++thread_id)
        {
            delete this->xoshiro_256_star_star[thread_id];
            this->xoshiro_256_star_star[thread_id] = NULL;
        }

        delete[] this->xoshiro_256_star_star;
        this->xoshiro_256_star_star = NULL;
    }
}


// ==========
// initialize
// ==========

/// \brief     Initializes an array of \c xoshiro_256_star_star objects.
///
/// \details   The size of the array is \c num_threads, corresponding to each
///            parallel thread. Also, the state of the i-th object is jumped
///            \c (i+1) times so that all random generators have diferent
///            start states. This is the main reason of using this class since
///            it aggregates multiple random generator objects, one for each
///            parallel thread, and all have different initial random state.
///
/// \param[in] seed
///            Seed for pseudo-random number generation. The same seed value is
///            used for all threads. If seed is negative integer, the given
///            seed value is ignored, and the processor time is used istead.

void RandomNumberGenerator::initialize(int64_t seed)
{
    assert(this->num_threads > 0);

    this->xoshiro_256_star_star = new Xoshiro256StarStar*[this->num_threads];

    for (int thread_id=0; thread_id < this->num_threads; ++thread_id)
    {
        this->xoshiro_256_star_star[thread_id] = new Xoshiro256StarStar(seed);

        // Repeate jump j times to have different initial state for each thread
        // This is the main purpose of this class.
        for (int j=0; j < thread_id+1; ++j)
        {
            this->xoshiro_256_star_star[thread_id]->jump();
        }
    }
}


// ====
// next
// ====

/// \brief     Generates the next random number in the sequence, depending on
///            the thread id.
///
/// \param[in] thread_id
///            The thread id of the parallel process.

uint64_t RandomNumberGenerator::next(const int thread_id)
{
    return this->xoshiro_256_star_star[thread_id]->next();
}
