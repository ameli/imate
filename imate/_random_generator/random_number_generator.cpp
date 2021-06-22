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

/// \brief Initializes with one parallel thread.
///

RandomNumberGenerator::RandomNumberGenerator()
{
    int num_threads_ = 1;
    this->initialize(num_threads_);
}


// =============
// Constructor 2
// =============

/// \brief Initializes with given number of parallel thread.
///

RandomNumberGenerator::RandomNumberGenerator(int num_threads_)
{
    this->initialize(num_threads_);
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
/// \param[in] num_threads_
///            Number of independent \c xoshiro_256_star_star objects to be
///            created for each parallel thread.

void RandomNumberGenerator::initialize(int num_threads_)
{
    assert(num_threads_ > 0);

    this->num_threads = num_threads_;
    this->xoshiro_256_star_star = new Xoshiro256StarStar[this->num_threads];

    for (int i=0; i < this->num_threads; ++i)
    {
        // Repeate jump j times to have different initial state for each thread
        // This is the main purpose of this class.
        for (int j=0; j < i+1; ++j)
        {
            this->xoshiro_256_star_star[i].jump();
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

uint64_t RandomNumberGenerator::next(int thread_id)
{
    return this->xoshiro_256_star_star[thread_id].next();
}
