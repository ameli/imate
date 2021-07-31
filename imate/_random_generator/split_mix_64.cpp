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
#include <cassert>  // assert
#include <ctime>  // std::time


// ===========
// Constructor
// ===========

/// \brief Constructor. Initializes the state with current time.
///

SplitMix64::SplitMix64()
{
    // std::time gives the second since epoch. This, if this function is called
    // multiple times a second, the std::time() results the same number. To
    // make it differ between each milliseconds, the std::clock is added, which
    // is the cpu time (in POSIX) or wall time (in windows) and in the unit of
    // system's clocks per second.
    uint64_t seed = static_cast<uint64_t>(std::time(0)) +
                    static_cast<uint64_t>(std::clock());

    // Seeding as follow only fills the first 32 bits of the 64-bit integer.
    // Repeat the first 32 bits on the second 32-bits to create a better 64-bit
    // random number
    this->state = (seed << 32) | seed;
}


// ===========
// Constructor
// ===========

/// \brief     Constructor. Initializes the state with an input integer.
///
/// \param[in] state_
///            A 64-bit integer to initialize the state. This number must be
///            non-zero.

SplitMix64::SplitMix64(uint64_t state_):
    state(state_)
{
    // Initial state must not be zero.
    assert(state_ != 0);
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
