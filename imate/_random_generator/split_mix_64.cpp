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

#include "./split_mix_64.h"
#include "./highres_time_stamp.h"  // get_highres_time_stamp
#include <cassert>  // assert


// ===========
// Constructor
// ===========

/// \brief Constructor. Initializes the state with current time.
///

SplitMix64::SplitMix64(const int64_t seed_)
{
    // Seed the random generating algorithm with a high resolution time counter
    uint64_t seed;

    if (seed_ >= 0)
    {
        seed = static_cast<uint64_t>(seed_);
    }
    else
    {
        // Negative integer is a flag to indicate using time to generate a seed
        seed = get_highres_time_stamp();
    }

    // Seeding as follow only fills the first 32 bits of the 64-bit integer.
    // Repeat the first 32 bits on the second 32-bits to create a better 64-bit
    // random number
    this->state = (seed << 32) | seed;
}


// ====
// next
// ====

/// \brief Generates the next presudo-random number in the sequence.
///

uint64_t SplitMix64::next()
{
    uint64_t z = (state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;

    return z ^ (z >> 31);
}
