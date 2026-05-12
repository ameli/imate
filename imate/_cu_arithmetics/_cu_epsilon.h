/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _CU_ARITHMETICS_CU_EPSILON_H_
#define _CU_ARITHMETICS_CU_EPSILON_H_

// =======
// Headers
// =======

#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __nv_bfloat16


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
///            epsilonition and multiplication of two float numbers.
///
/// \sa        cu_arithmetics::add,
///            cu_arithmetics::is_equal
///            cu_arithmetics::cast

namespace cu_arithmetics
{
    // =======
    // epsilon
    // =======

    /// \brief     epsilon for various floating point precisions.
    ///
    /// \note      This is a \c __host__ function only, since
    ///            \c std::numeric_limits cannot be called from a \c __device__
    ///            code.
    ///
    /// \return    eps
    ///            Epsilon
    ///
    /// \sa        cu_arithmetics::is_equal

    template <typename DataType>
    inline __host__ __device__ DataType epsilon();


    // =======
    // epsilon (__nv_fp8_e5m2)
    // =======

    /// \brief     epsilon for \c __nv_fp8_e5m2 type, which is equal to
    ///            \f$ 2^{-2} \f$ since \c __nv_fp8_e5m2 has 2 bits for
    ///            fraction (mantissa).
    ///
    /// \return    eps
    ///            Epsilon
    ///
    /// \sa        cu_arithmetics::is_equal
    
    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e5m2 epsilon<__nv_fp8_e5m2>()
    {
        // This is 2^{-10}, as __nv_fp8_e5m2 type has 2 digits for mantissa. 
        return __nv_fp8_e5m2(0.25f);
    }
    #endif


    // =======
    // epsilon (__nv_fp8_e4m3)
    // =======

    /// \brief     epsilon for \c __nv_fp8_e4m3 type, which is equal to
    ///            \f$ 2^{-3} \f$ since \c __nv_fp8_e4m3 has 3 bits for
    ///            fraction (mantissa).
    ///
    /// \return    eps
    ///            Epsilon
    ///
    /// \sa        cu_arithmetics::is_equal
    
    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e4m3 epsilon<__nv_fp8_e4m3>()
    {
        // This is 2^{-10}, as __nv_fp8_e4m3 type has 3 digits for mantissa. 
        return __nv_fp8_e4m3(0.125f);
    }
    #endif


    // =======
    // epsilon (__half)
    // =======

    /// \brief     epsilon for \c __half type, which is equal to \f$ 2^{-10}
    ///            \f$ since \c __half has 10 bits for fraction (mantissa).
    ///
    /// \return    eps
    ///            Epsilon
    ///
    /// \sa        cu_arithmetics::is_equal
    
    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    inline __host__ __device__ __half epsilon<__half>()
    {
        // This is 2^{-10}, as __half type has 10 digits for mantissa. 
        return __float2half(0.00097656f);
    }
    #endif


    // =======
    // epsilon (__nv_bfloat16)
    // =======

    /// \brief     epsilon for \c __nv_bfloat16 type, which is equal to
    ///            \f$ 2^{-7} \f$ since \c __nv_bfloat16 has 7 bits for
    ///            fraction (mantissa).
    ///
    /// \return    eps
    ///            Epsilon
    ///
    /// \sa        cu_arithmetics::is_equal
    
#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    inline __host__ __device__ __nv_bfloat16 epsilon<__nv_bfloat16>()
    {
        // This is 2^{-7}, as __half type has 10 digits for mantissa. 
        return __float2bfloat16(0.0078125f);
    }
    #endif


    // =======
    // epsilon (float)
    // =======

    /// \brief     epsilon for \c float type, which is equal to \f$ 2^{-23}
    ///            \f$ since \c float has 23 bits for fraction (mantissa).
    ///
    /// \return    eps
    ///            Epsilon
    ///
    /// \sa        cu_arithmetics::is_equal
    
    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    inline __host__ __device__ float epsilon<float>()
    {
        // Instead of the hard-coded number below, one may use
        // the std::numeric_limits::epsilon(), but, then this function cannot
        // be called as a __device__ code. Hence, the value of 2^(-23) is hard
        // coded below.
        return 1.1920929e-7f;
    }
    #endif


    // =======
    // epsilon (float)
    // =======

    /// \brief     epsilon for \c float type, which is equal to \f$ 2^{-52}
    ///            \f$ since \c float has 52 bits for fraction (mantissa).
    ///
    /// \return    eps
    ///            Epsilon
    ///
    /// \sa        cu_arithmetics::is_equal
    
    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    inline __host__ __device__ double epsilon<double>()
    {
        // Instead of the hard-coded number below, one may use
        // the std::numeric_limits::epsilon(), but, then this function cannot
        // be called as a __device__ code. Hence, the value of 2^(-52) is hard
        // coded below.
        return 2.220446049250313e-16;
    }
    #endif

}  // namespace cu_arithmetics

#endif  // _CU_ARITHMETICS_CU_EPSILON_H_
