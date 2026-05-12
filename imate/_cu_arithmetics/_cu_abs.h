/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _CU_ARITHMETICS_CU_ABS_H_
#define _CU_ARITHMETICS_CU_ABS_H_

// =======
// Headers
// =======

#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __nv_bfloat16, __habs

#include <cmath>  // std::abs
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
/// \sa        cu_arithmetics::add,
///            cu_arithmetics::is_equal
///            cu_arithmetics::cast

namespace cu_arithmetics
{
    // ===
    // abs
    // ===

    /// \brief     Absolute value of a floating point number.
    ///
    /// \param[in] x
    ///            Operand.
    /// \return    y
    ///            Absolute value of \c x
    ///
    /// \sa        cu_arithmetics::is_equal

    template <typename DataType>
    inline __host__ __device__ DataType abs(const DataType x);


    // ===
    // abs (__nv_fp8_e5m2)
    // ===

    /// \brief     Absolute value of a floating point number in
    ///            \c __nv_fp8_e5m2 type.
    ///
    /// \param[in] x
    ///            Operand.
    /// \return    y
    ///            Absolute value of \c x
    ///
    /// \sa        cu_arithmetics::is_equal
    
    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e5m2 abs<__nv_fp8_e5m2>(
            const __nv_fp8_e5m2 x)
    {
        // Not implemented
        assert(false);

        return __nv_fp8_e5m2(NAN);
    }
    #endif


    // ===
    // abs (__nv_fp8_e4m3)
    // ===

    /// \brief     Absolute value of a floating point number in
    ///            \c __nv_fp8_e4m3 type.
    ///
    /// \param[in] x
    ///            Operand.
    /// \return    y
    ///            Absolute value of \c x
    ///
    /// \sa        cu_arithmetics::is_equal
    
    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e4m3 abs<__nv_fp8_e4m3>(
            const __nv_fp8_e4m3 x)
    {
        // Not implemented
        assert(false);
        
        return __nv_fp8_e4m3(NAN);
    }
    #endif


    // ===
    // abs (__half)
    // ===

    /// \brief     Absolute value of a floating point number in \c __half type.
    ///
    /// \param[in] x
    ///            Operand.
    /// \return    y
    ///            Absolute value of \c x
    ///
    /// \sa        cu_arithmetics::is_equal
    
    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    inline __host__ __device__ __half abs<__half>(const __half x)
    {
        return __habs(x);
    }
    #endif


    // ===
    // abs (__nv_bfloat16)
    // ===

    /// \brief     Absolute value of a floating point number in
    ///            \c __nv_bfloat16 type.
    ///
    /// \param[in] x
    ///            Operand.
    /// \return    y
    ///            Absolute value of \c x
    ///
    /// \sa        cu_arithmetics::is_equal

    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    inline __host__ __device__ __nv_bfloat16 abs<__nv_bfloat16>(
            const __nv_bfloat16 x)
    {
        return __habs(x);
    }
    #endif


    // ===
    // abs (float)
    // ===

    /// \brief     Absolute value of a floating point number in \c float type.
    ///
    /// \param[in] x
    ///            Operand.
    /// \return    y
    ///            Absolute value of \c x
    ///
    /// \sa        cu_arithmetics::is_equal
 
    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    inline __host__ __device__ float abs<float>(const float x)
    {
        return std::abs(x);
    }
    #endif


    // ===
    // abs (double)
    // ===

    /// \brief     Absolute value of a floating point number in \c double type.
    ///
    /// \param[in] x
    ///            Operand.
    /// \return    y
    ///            Absolute value of \c x
    ///
    /// \sa        cu_arithmetics::is_equal
    
    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    inline __host__ __device__ double abs<double>(const double x)
    {
        return std::abs(x);
    }
    #endif

}  // namespace cu_arithmetics

#endif  // _CU_ARITHMETICS_CU_ABS_H_
