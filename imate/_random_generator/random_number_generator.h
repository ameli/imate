/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _RANDOM_GENERATOR_RANDOM_NUMBER_GENERATOR_H_
#define _RANDOM_GENERATOR_RANDOM_NUMBER_GENERATOR_H_


// =======
// Headers
// =======

#include <stdint.h>  // int64_t, uint64_t
class Xoshiro256StarStar;


// =======================
// Random Number Generator
// =======================

/// \class   RandomNumberGenerator
///
/// \brief   Generates 64-bit integers on multiple parallel threads.
///
/// \details This class creates multiple instances of \c Xoshiro256StarStar
///          class, with the number of parallel threads. On each thread, it
///          generates random integers independently.
///
///          The necessity of using this class is when all threads start at the
///          same time, hence the initial seeds cannot be distinguished using
///          the \c time() function (since time the same for all threads). Thus
///          the random generator in all threads generate the same senquence of
///          numbers. To avoid this issue, this class jumps the sequence of
///          each instance of \c Xoshiro256StarStar. Thus, even if their
///          initial state seed is the same, the jump in the state variable
///          makes it different by 2^128 on the sequence.
///
///          All member variables of this class aer static and they can be used
///          without declaring an instance of this class. Also, in the same
///          thread, if we call this class from multiple functions, the same
///          sequence if used since the internal state variable is static.
///
///          This class can be used by either
///
///          1. Declare a new instance of this class. In this case, the
///             destructor will remove the variables at the end of its lifetime
///             once the last instance of this class goes out of scope.
///
///                 num_threads = omp_get_num_threads();
///
///                 // Declare an instance for num_thread parallel threads
///                 RandomNumberGenerator random_number_generator(num_threads);
///
///                 // Generate random numbers
///                 #pragma omp parallel
///                 {
///                     int tid = omp_get_thread_num();
///
///                     #pragma omp for
///                     for (int i = 0; i < n; ++i)
///                     {
///                         uint64_t a = RandomNumberGenerator::next(tid);
///                     }
///                 }
///
///                 // The random_number_generator will go out of scope and it
///                 // automatically deallocate internal static arrays.
///
///
///          2. Call all functions sttaically. In this case, the internal
///             static arrays should be deallocated manually. Such as
///
///                 num_threads = omp_get_num_threads();
///
///                 // Create an internal array of random generator per thread
///                 RandomNumberGenerator::initialize(num_threads);
///
///                 // Generate random numbers
///                 #pragma omp parallel
///                 {
///                     int tid = omp_get_thread_num();
///
///                     #pragma omp for
///                     for (int i = 0; i < n; ++i)
///                     {
///                         uint64_t a = RandomNumberGenerator::next(tid);
///                     }
///                 }
///
///                 // Deallcoate internal static arrays of this class
///                 RandomNumberGenerator::deallocate();
///
/// \sa      RandomArrayGenerator,
///          Xoshiro256StarStar

class RandomNumberGenerator
{
    public:
        // Member methods
        RandomNumberGenerator();
        explicit RandomNumberGenerator(const int num_threads_);
        explicit RandomNumberGenerator(
                const int num_threads_,
                const int64_t seed);
        ~RandomNumberGenerator();
        uint64_t next(const int thread_id);

    protected:
        // Member methods
        void initialize(int64_t seed);

        // Member data
        const int num_threads;
        Xoshiro256StarStar** xoshiro_256_star_star;
};


#endif  // _RANDOM_GENERATOR_RANDOM_NUMBER_GENERATOR_H_
