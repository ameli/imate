/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _RANDOM_GENERATOR_XOSHIRO_256_STAR_STAR_H_
#define _RANDOM_GENERATOR_XOSHIRO_256_STAR_STAR_H_


// =======
// Headers
// =======

#include <stdint.h>  // uint64_t, UINT64_C


// =====================
// Xoshiro 256 Star Star
// =====================

/// \class   Xoshiro256StarStar
///
/// \brief   Pseudo-random integer generator. This class generates 64-bit
///          integer using Xoshiro256** algorithm.
///
/// \details The Xoshiro256** algorithm has 256-bit state space, and passes all
///          statistical tests, including the BigCrush. The state of this class
///          is initialized using \c SplitMix64 random generator.
///
///          A very similar method to Xoshiro256** is Xoshiro256++ which has
///          the very same properties and speed as the Xoshiro256**. An
///          alternative method is Xoshiro256+, which is 15% faster, but it
///          suffers linear dependency of the lower 4 bits. It is usually used
///          for generating floating numbers using the upper 53 bits and
///          discard the lower bits.<F3>
///
///          The Xoshiro256** algorithm is develped by David Blackman and
///          Sebastiano Vigna (2018) and the source code can be found at:
///          https://prng.di.unimi.it/xoshiro256starstar.c
///
/// \sa      SplitMix64

class Xoshiro256StarStar
{
    public:
        Xoshiro256StarStar();
        ~Xoshiro256StarStar();
        uint64_t next();
        void jump();
        void long_jump();

    protected:
        static inline uint64_t rotation_left(
                const uint64_t x,
                int k);

        // Member data
        uint64_t *state;
};


#endif  // _RANDOM_GENERATOR_XOSHIRO_256_STAR_STAR_H_
