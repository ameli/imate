/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _CU_ARITHMETICS_CU_MUL_H_
#define _CU_ARITHMETICS_CU_MUL_H_

// =======
// Headers
// =======

#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __nv_bfloat16, __hmul
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
///            addition and multiplication of two float numbers.
///
/// \sa        cu_arithmetics::div,
///            cu_arithmetics::is_equal
///            cu_arithmetics::cast

namespace cu_arithmetics
{
    // ===
    // mul
    // ===

    /// \brief     Multiply two floating point numbers in
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
    inline __host__ __device__ DataType mul(
            const DataType x,
            const DataType y);


    // ===
    // mul (__nv_fp8_e5m2)
    // ===

    /// \brief     Multiply two \c __nv_fp8_e5m2 type numbers in
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
    inline __host__ __device__ __nv_fp8_e5m2 mul<__nv_fp8_e5m2>(
            const __nv_fp8_e5m2 x,
            const __nv_fp8_e5m2 y)
    {
        // Not implemented
        assert(false);

        return __nv_fp8_e5m2(NAN);
    }
    #endif


    // ===
    // mul (__nv_fp8_e4m3)
    // ===

    /// \brief     Multiply two \c __nv_fp8_e4m3 type numbers in
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
    inline __host__ __device__ __nv_fp8_e4m3 mul<__nv_fp8_e4m3>(
            const __nv_fp8_e4m3 x,
            const __nv_fp8_e4m3 y)
    {
        // Not implemented
        assert(false);

        return __nv_fp8_e4m3(NAN);
    }
    #endif


    // ===
    // mul (__half)
    // ===

    /// \brief     Multiply two \c __half type numbers in
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
    inline __host__ __device__ __half mul<__half>(
            const __half x,
            const __half y)
    {
        return __hmul(x, y);
    }
    #endif


    // ===
    // mul (__nv_bfloat16)
    // ===

    /// \brief     Multiply two \c __nv_bfloat16 type numbers in
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
    inline __host__ __device__ __nv_bfloat16 mul<__nv_bfloat16>(
            const __nv_bfloat16 x,
            const __nv_bfloat16 y)
    {
        return __hmul(x, y);
    }
    #endif


    // ===
    // mul (float)
    // ===

    /// \brief     Multiply two \c float type numbers in
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
    inline __host__ __device__ float mul<float>(
            const float x,
            const float y)
    {
        return x * y;
    }
    #endif


    // ===
    // mul (double)
    // ===

    /// \brief     Multiply two \c double type numbers in
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
    inline __host__ __device__ double mul<double>(
            const double x,
            const double y)
    {
        return x * y;
    }
    #endif


    // ===
    // mul
    // ===

    /// \brief     Multiply three floating point numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \param[in] z
    ///            Third operand.
    /// \return    w
    ///            Product of \c x, \c y, and \c z.
    ///
    /// \sa        cu_arithmetics::div
    
    template <typename DataType>
    inline __host__ __device__ DataType mul(
            const DataType x,
            const DataType y,
            const DataType z);


    // ===
    // mul (__half)
    // ===

    /// \brief     Multiply three \c __half numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \param[in] z
    ///            Third operand.
    /// \return    w
    ///            Product of \c x, \c y, and \c z.
    ///
    /// \sa        cu_arithmetics::div
    
    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    inline __host__ __device__ __half mul<__half>(
            const __half x,
            const __half y,
            const __half z)
    {
        return __hmul(__hmul(x, y), z);
    }
    #endif


    // ===
    // mul (__nv_bfloat16)
    // ===

    /// \brief     Multiply three \c __nv_bfloat16 numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \param[in] z
    ///            Third operand.
    /// \return    w
    ///            Product of \c x, \c y, and \c z.
    ///
    /// \sa        cu_arithmetics::div
    
    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    inline __host__ __device__ __nv_bfloat16 mul<__nv_bfloat16>(
            const __nv_bfloat16 x,
            const __nv_bfloat16 y,
            const __nv_bfloat16 z)
    {
        return __hmul(__hmul(x, y), z);
    }
    #endif


    // ===
    // mul (float)
    // ===

    /// \brief     Multiply three \c float numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \param[in] z
    ///            Third operand.
    /// \return    w
    ///            Product of \c x, \c y, and \c z.
    ///
    /// \sa        cu_arithmetics::div
    
    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    inline __host__ __device__ float mul<float>(
            const float x,
            const float y,
            const float z)
    {
        return x * y * z;
    }
    #endif


    // ===
    // mul (double)
    // ===

    /// \brief     Multiply three \c double numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \param[in] z
    ///            Third operand.
    /// \return    w
    ///            Product of \c x, \c y, and \c z.
    ///
    /// \sa        cu_arithmetics::div
    
    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    inline __host__ __device__ double mul<double>(
            const double x,
            const double y,
            const double z)
    {
        return x * y * z;
    }
    #endif

}  // namespace cu_arithmetics

#endif  // _CU_ARITHMETICS_CU_MUL_H_
