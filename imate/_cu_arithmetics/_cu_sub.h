/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _CU_ARITHMETICS_CU_SUB_H_
#define _CU_ARITHMETICS_CU_SUB_H_

// =======
// Headers
// =======

#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __nv_bfloat16, __hsub
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
///            subition and multiplication of two float numbers.
///
/// \sa        cu_arithmetics::add,
///            cu_arithmetics::is_equal
///            cu_arithmetics::cast

namespace cu_arithmetics
{
    // ===
    // sub
    // ===

    /// \brief     Subtract two floating point numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Sum of \c x and \c y
    ///
    /// \sa        cu_arithmetics::add

    template <typename DataType>
    inline __host__ __device__ DataType sub(
            const DataType x,
            const DataType y);


    // ===
    // sub (__nv_fp8_e5m2)
    // ===

    /// \brief     Subtract two \c __nv_fp8_e5m2 type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Sum of \c x and \c y
    ///
    /// \sa        cu_arithmetics::add
    
    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e5m2 sub<__nv_fp8_e5m2>(
            const __nv_fp8_e5m2 x,
            const __nv_fp8_e5m2 y)
    {
        // Not implemented
        assert(false);

        return __nv_fp8_e5m2(NAN);
    }
    #endif


    // ===
    // sub (__nv_fp8_e4m3)
    // ===

    /// \brief     Subtract two \c __nv_fp8_e4m3 type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Sum of \c x and \c y
    ///
    /// \sa        cu_arithmetics::add
    
    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e4m3 sub<__nv_fp8_e4m3>(
            const __nv_fp8_e4m3 x,
            const __nv_fp8_e4m3 y)
    {
        // Not implemented
        assert(false);

        return __nv_fp8_e4m3(NAN);
    }
    #endif


    // ===
    // sub (__half)
    // ===

    /// \brief     Subtract two \c __half type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Sum of \c x and \c y
    ///
    /// \sa        cu_arithmetics::add
    
    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    inline __host__ __device__ __half sub<__half>(
            const __half x,
            const __half y)
    {
        return __hsub(x, y);
    }
    #endif


    // ===
    // sub (__nv_bfloat16)
    // ===

    /// \brief     Subtract two \c __nv_bfloat16 type numbers in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Sum of \c x and \c y
    ///
    /// \sa        cu_arithmetics::add
    
    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    inline __host__ __device__ __nv_bfloat16 sub<__nv_bfloat16>(
            const __nv_bfloat16 x,
            const __nv_bfloat16 y)
    {
        return __hsub(x, y);
    }
    #endif


    // ===
    // sub (float)
    // ===

    /// \brief     Subtract two \c float type numbers.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Sum of \c x and \c y
    ///
    /// \sa        cu_arithmetics::add
    
    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    inline __host__ __device__ float sub<float>(
            const float x,
            const float y)
    {
        return x - y;
    }
    #endif


    // ===
    // sub (double)
    // ===

    /// \brief     Subtract two \c double type float numbers.
    ///
    /// \param[in] x
    ///            First operand.
    /// \param[in] y
    ///            Second operand.
    /// \return    z
    ///            Sum of \c x and \c y
    ///
    /// \sa        cu_arithmetics::add
    
    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    inline __host__ __device__ double sub<double>(
            const double x,
            const double y)
    {
        return x - y;
    } 
    #endif

}  // namespace cu_arithmetics

#endif  // _CU_ARITHMETICS_CU_SUB_H_
