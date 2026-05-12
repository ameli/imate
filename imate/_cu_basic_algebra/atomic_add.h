/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CU_BASIC_ALGEBRA_ATOMIC_ADD_H_
#define _CU_BASIC_ALGEBRA_ATOMIC_ADD_H_

// =======
// Headers
// =======

#include <cuda_runtime.h>


// ==========
// atomic add
// ==========

/// \brief Definition of \c atomic_add for pre-Pascla Nvidia archetectures.
///
/// \details \c atomicAdd for \c double precision floating-point numbers is not
///          available on devices with compute capability lower than 6.0 but it
///          can be implemented as given in this function.

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = \
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(
                address_as_ull,
                assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN
    // (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

#endif  // _CU_BASIC_ALGEBRA_ATOMIC_ADD_H_
