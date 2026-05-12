/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _CU_ARITHMETICS_CU_DIV_H_
#define _CU_ARITHMETICS_CU_DIV_H_

// =======
// Headers
// =======

#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __nv_bfloat16, __hdiv
#include <cassert>  // assert


// =============
// cu arithmetic
// =============

/// \namespace cu_arithmetic
///
/// \brief     perform arithmetics for \c __half and \c __nv_bfloat16 types
///            in round-to-nearest-even mode.
///
/// \details   This namespace is a templated unifying API for CUDA's arithmetic
///            operations for \c __half and \c __nv_bfloat16 types, including
///            addition and divtiplication of two float numbers.
///
/// \sa        cu_arithmetics::mul,
///            cu_arithmetics::is_equal
///            cu_arithmetics::cast

namespace cu_arithmetics
{
    // ===
    // div
    // ===

    /// \brief     Divide two floating point numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Product of \c x and \c y
    ///
    /// \sa        cu_arithmetics::div
    
    template <typename DataType>
    inline __host__ __device__ DataType div(
            const DataType x,
            const DataType y);


    // ===
    // div (__nv_fp8_e5m2)
    // ===

    /// \brief     Divide two \c __nv_fp8_e5m2 type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Product of \c x and \c y
    ///
    /// \sa        cu_arithmetics::div
    
    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e5m2 div<__nv_fp8_e5m2>(
            const __nv_fp8_e5m2 x,
            const __nv_fp8_e5m2 y)
    {
        // Not implemented
        assert(false);

        return __nv_fp8_e5m2(NAN);
    }
    #endif


    // ===
    // div (__nv_fp8_e4m3)
    // ===

    /// \brief     Divide two \c __nv_fp8_e4m3 type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Product of \c x and \c y
    ///
    /// \sa        cu_arithmetics::div
    
    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e4m3 div<__nv_fp8_e4m3>(
            const __nv_fp8_e4m3 x,
            const __nv_fp8_e4m3 y)
    {
        // Not implemented
        assert(false);

        return __nv_fp8_e4m3(NAN);
    }
    #endif


    // ===
    // div (__half)
    // ===

    /// \brief     Divide two \c __half type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Product of \c x and \c y
    ///
    /// \sa        cu_arithmetics::div
    
    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    inline __host__ __device__ __half div<__half>(
            const __half x,
            const __half y)
    {
        return __hdiv(x, y);
    }
    #endif


    // ===
    // div (__nv_bfloat16)
    // ===

    /// \brief     Divide two \c __nv_bfloat16 type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Product of \c x and \c y
    ///
    /// \sa        cu_arithmetics::div
    
    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    inline __host__ __device__ __nv_bfloat16 div<__nv_bfloat16>(
            const __nv_bfloat16 x,
            const __nv_bfloat16 y)
    {
        return __hdiv(x, y);
    }
    #endif


    // ===
    // div (float)
    // ===

    /// \brief     Divide two \c float type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Product of \c x and \c y
    ///
    /// \sa        cu_arithmetics::div
    
    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    inline __host__ __device__ float div<float>(
            const float x,
            const float y)
    {
        return x / y;
    }
    #endif


    // ===
    // div (double)
    // ===

    /// \brief     Divide two \c double type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Product of \c x and \c y
    ///
    /// \sa        cu_arithmetics::div
    
    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    inline __host__ __device__ double div<double>(
            const double x,
            const double y)
    {
        return x / y;
    }
    #endif

}  // namespace cu_arithmetics

#endif  // _CU_ARITHMETICS_CU_DIV_H_
