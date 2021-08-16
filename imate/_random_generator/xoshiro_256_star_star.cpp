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

#include "./xoshiro_256_star_star.h"
#include <cstdlib>  // NULL
#include "./split_mix_64.h"  // SplitMix64

// stdint.h in old compilers (e.g. gcc 4.4.7) does not declare UINT64_C macro.
#ifndef UINT64_C
    #define UINT64_C(c) static_cast<uint64_t>(c)
#endif


// ===========
// Constructor
// ===========

/// \brief Constructor. It initializes the state variable with random integers
///        using \c splitmix64 pseudo-random generator.
///
Xoshiro256StarStar::Xoshiro256StarStar():
    state(NULL)
{
    // Allocate state
    this->state = new uint64_t[4];

    // Initializing SplitMix64 random generator
    SplitMix64 split_mix_64;

    for (int i=0; i < 4; ++i)
    {
        this->state[i] = split_mix_64.next();
    }
}


// ==========
// Destructor
// ==========

/// \brief Destructor.
///

Xoshiro256StarStar::~Xoshiro256StarStar()
{
    if (this->state != NULL)
    {
        delete[] this->state;
        this->state = NULL;
    }
}


// ====
// next
// ====

/// \brief Generates the next presudo-random number.
///

uint64_t Xoshiro256StarStar::next()
{
    const uint64_t result = this->rotation_left(this->state[1] * 5, 7) * 9;

    const uint64_t t = this->state[1] << 17;

    this->state[2] ^= this->state[0];
    this->state[3] ^= this->state[1];
    this->state[1] ^= this->state[2];
    this->state[0] ^= this->state[3];

    this->state[2] ^= t;

    this->state[3] = this->rotation_left(this->state[3], 45);

    return result;
}


// ====
// jump
// ====

/// \brief Jump function for the generator. It is equivalent to 2^128 calls to
///        \c next(); it can be used to generate 2^128 non-overlapping
///        subsequences for parallel computations.

void Xoshiro256StarStar::jump()
{
    static const uint64_t JUMP[] = {
        0x180ec6d33cfd0aba,
        0xd5a61266f0c9392c,
        0xa9582618e03fc9aa,
        0x39abdc4529b1661c};

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t s3 = 0;

    for (unsigned int i = 0; i < sizeof(JUMP) / sizeof(*JUMP); ++i)
    {
        for (int b = 0; b < 64; ++b)
        {
            if (JUMP[i] & UINT64_C(1) << b)
            {
                s0 ^= this->state[0];
                s1 ^= this->state[1];
                s2 ^= this->state[2];
                s3 ^= this->state[3];
            }

            this->next();
        }
    }

    this->state[0] = s0;
    this->state[1] = s1;
    this->state[2] = s2;
    this->state[3] = s3;
}


// =========
// long jump
// =========

/// \brief  Long jump function for the generator. It is equivalent to 2^192
///         calls to \c next(). It can be used to generate 2^64 starting points
///         from each of which \c jump() will generate 2^64 non-overlapping
///         subsequences for parallel distributed computations.

void Xoshiro256StarStar::long_jump()
{
    static const uint64_t LONG_JUMP[] = {
        0x76e15d3efefdcbbf,
        0xc5004e441c522fb3,
        0x77710069854ee241,
        0x39109bb02acbe635};

    uint64_t s0 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t s3 = 0;

    for (unsigned int i = 0; i < sizeof(LONG_JUMP) / sizeof(*LONG_JUMP); ++i)
    {
        for (int b = 0; b < 64; ++b)
        {
            if (LONG_JUMP[i] & UINT64_C(1) << b)
            {
                s0 ^= this->state[0];
                s1 ^= this->state[1];
                s2 ^= this->state[2];
                s3 ^= this->state[3];
            }

            this->next();
        }
    }

    this->state[0] = s0;
    this->state[1] = s1;
    this->state[2] = s2;
    this->state[3] = s3;
}


// ===========
// rotate left
// ===========

/// \brief     Rotates the bits of a 64 bit integer toward left.
/// \param[in] x
///            A 64 bit integer to rotate its bits toward left.
/// \param[in] k
///            Number of bit places to rotate each bit of \c x
/// \return    A 64 bit integer where each bit is \c k bit moved left

inline uint64_t Xoshiro256StarStar::rotation_left(
        const uint64_t x,
        int k)
{
    return (x << k) | (x >> (64 - k));
}
