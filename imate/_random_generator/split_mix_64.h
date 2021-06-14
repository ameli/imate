/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _RANDOM_GENERATOR_SPLIT_MIX_64_H_
#define _RANDOM_GENERATOR_SPLIT_MIX_64_H_


// ======
// Header
// ======

#include <stdint.h>  // uint64_t


// ============
// Split Mix 64
// ============

/// \class SplitMix64
///
/// \brief   Pseudo-random integer generator. This class generates 64-bit
///          integer using SplitMix64 algorithm.
///
/// \details The SplitMix64 algorithm is very fast but does not pass all
///          statistical tests. This class is primarily used to initialize the
///          states of the \c Xoshiro256StarStar class.
///
///          The SplitMix64 algorithm is develped by Sebastiano Vigna (2015)
///          and the source code is available at:
///          https://prng.di.unimi.it/splitmix64.c
///
/// \sa      Xoshiro256StarStar

class SplitMix64
{
    public:
        SplitMix64();
        explicit SplitMix64(uint64_t state_);
        uint64_t next();

    protected:
        uint64_t state;
};

#endif  // _RANDOM_GENERATOR_SPLIT_MIX_64_H_
