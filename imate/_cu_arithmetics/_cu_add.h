/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _CU_ARITHMETICS_CU_ADD_H_
#define _CU_ARITHMETICS_CU_ADD_H_

// =======
// Headers
// =======

#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __nv_bfloat16, __hadd
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
/// \sa        cu_arithmetics::sub,
///            cu_arithmetics::is_equal
///            cu_arithmetics::cast

namespace cu_arithmetics
{
    // ===
    // add
    // ===

    /// \brief     Add two floating point numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Sum of \c x and \c y
    ///
    /// \sa        cu_arithmetics::sub

    template <typename DataType>
    inline __host__ __device__ DataType add(
            const DataType x,
            const DataType y);


    // ===
    // add (__nv_fp8_e5m2)
    // ===

    /// \brief     Add two \c __nv_fp8_e5m2 type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Sum of \c x and \c y
    ///
    /// \sa        cu_arithmetics::sub
    
    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e5m2 add<__nv_fp8_e5m2>(
            const __nv_fp8_e5m2 x,
            const __nv_fp8_e5m2 y)
    {
        // Not implemented
        assert(false);

        return __nv_fp8_e5m2(NAN);
    }
    #endif


    // ===
    // add (__nv_fp8_e4m3)
    // ===

    /// \brief     Add two \c __nv_fp8_e4m3 type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Sum of \c x and \c y
    ///
    /// \sa        cu_arithmetics::sub
    
    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e4m3 add<__nv_fp8_e4m3>(
            const __nv_fp8_e4m3 x,
            const __nv_fp8_e4m3 y)
    {
        // Not implemented
        assert(false);

        return __nv_fp8_e4m3(NAN);
    }
    #endif


    // ===
    // add (__half)
    // ===

    /// \brief     Add two \c __half type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Sum of \c x and \c y
    ///
    /// \sa        cu_arithmetics::sub
    
    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    inline __host__ __device__ __half add<__half>(
            const __half x,
            const __half y)
    {
        return __hadd(x, y);
    }
    #endif


    // ===
    // add (__nv_bfloat16)
    // ===

    /// \brief     Add two \c __nv_bfloat16 type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Sum of \c x and \c y
    ///
    /// \sa        cu_arithmetics::sub
    
    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    inline __host__ __device__ __nv_bfloat16 add<__nv_bfloat16>(
            const __nv_bfloat16 x,
            const __nv_bfloat16 y)
    {
        return __hadd(x, y);
    }
    #endif


    // ===
    // add (float)
    // ===

    /// \brief     Add two \c float type numbers.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Sum of \c x and \c y
    ///
    /// \sa        cu_arithmetics::sub
    
    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    inline __host__ __device__ float add<float>(
            const float x,
            const float y)
    {
        return x + y;
    }
    #endif


    // ===
    // add (double)
    // ===

    /// \brief     Add two \c double type float numbers.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Sum of \c x and \c y
    ///
    /// \sa        cu_arithmetics::sub
    
    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    inline __host__ __device__ double add<double>(
            const double x,
            const double y)
    {
        return x + y;
    }
    #endif


    // ===
    // add
    // ===

    /// \brief     Add three floating point numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \param[in] z
    ///            Third operand.
    /// \return    w
    ///            Sum of \c x, \c y, and \c z.
    ///
    /// \sa        cu_arithmetics::sub

    template <typename DataType>
    inline __host__ __device__ DataType add(
            const DataType x,
            const DataType y,
            const DataType z);


    // ===
    // add (__half)
    // ===

    /// \brief     Add three \c __half type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \param[in] z
    ///            Third operand.
    /// \return    w
    ///            Sum of \c x, \c y, and \c z.
    ///
    /// \sa        cu_arithmetics::sub
    
    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    inline __host__ __device__ __half add<__half>(
            const __half x,
            const __half y,
            const __half z)
    {
        return __hadd(__hadd(x, y), z);
    }
    #endif


    // ===
    // add (__nv_bfloat16)
    // ===

    /// \brief     Add three \c __nv_bfloat16 type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \param[in] z
    ///            Third operand.
    /// \return    w
    ///            Sum of \c x, \c y, and \c z.
    ///
    /// \sa        cu_arithmetics::sub
    
    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    inline __host__ __device__ __nv_bfloat16 add<__nv_bfloat16>(
            const __nv_bfloat16 x,
            const __nv_bfloat16 y,
            const __nv_bfloat16 z)
    {
        return __hadd(__hadd(x, y), z);
    }
    #endif


    // ===
    // add (float)
    // ===

    /// \brief     Add three \c float type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \param[in] z
    ///            Third operand.
    /// \return    w
    ///            Sum of \c x, \c y, and \c z.
    ///
    /// \sa        cu_arithmetics::sub
    
    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    inline __host__ __device__ float add<float>(
            const float x,
            const float y,
            const float z)
    {
        return x + y + z;
    }
    #endif


    // ===
    // add (double)
    // ===

    /// \brief     Add three \c double type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \param[in] z
    ///            Third operand.
    /// \return    w
    ///            Sum of \c x, \c y, and \c z.
    ///
    /// \sa        cu_arithmetics::sub
    
    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    inline __host__ __device__ double add<double>(
            const double x,
            const double y,
            const double z)
    {
        return x + y + z;
    }
    #endif

}  // namespace cu_arithmetics

#endif  // _CU_ARITHMETICS_CU_ADD_H_
