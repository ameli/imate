/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _CU_ARITHMETICS_CU_CAST_H_
#define _CU_ARITHMETICS_CU_CAST_H_


// =======
// Headers
// =======

#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __half2float, __float2half,
                                         // __int2half_rn, __uint2half,
                                         // __ll2half, __ull2half
                                         // __nv_bfloat16, __bfloat162float,
                                         // __float2bfloat16,
                                         // __int2bfloat16_rn, __uint2bfloat16,
                                         // __ll2bfloat16, __ull2bfloat16


// ==============
// cu arithmetics
// ==============

/// \namespace cu_arithmetics
///
/// \brief     Cast from \c float to \c __half and \c __nv_bfloat16 types and
///            vice-versa, and \c float to \c double and vice-versa.
///
/// \details   This namespace is a templated unifying API casting CUDA's
///            types, including \c __half and \c __nv_bfloat16 types to
///            \c float and vice-versa and \c float to \c double and
///            vice-versa.
///
/// \sa        cu_arithmetics::add,
///            cu_arithmetics::mul,
///            cu_arithmetics::is_equal

namespace cu_arithmetics
{
    // ====
    // cast
    // ====

    /// \brief     Cast a floating point type to another floating point type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    x
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul

    template <typename InputDataType, typename OutputDataType>
    inline __host__ __device__ OutputDataType cast(const InputDataType x);


    // ====
    // cast (__nv_fp8_e5m2 to float)
    // ====

    /// \brief     Cast \c __nv_fp8_e5m2 type to \c float type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    inline __host__ __device__ float cast<__nv_fp8_e5m2, float>(
            const __nv_fp8_e5m2 x)
    {
        return float(x);
    }
    #endif


    // ====
    // cast (float to __nv_fp8_e5m2)
    // ====

    /// \brief     Cast \c float type to \c __nv_fp8_e5m2 type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e5m2 cast<float, __nv_fp8_e5m2>(
            const float x)
    {
        return __nv_fp8_e5m2(x);
    }
    #endif


    // ====
    // cast (__nv_fp8_e5m2 to double)
    // ====

    /// \brief     Cast \c __nv_fp8_e5m2 type to \c double type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    inline __host__ __device__ double cast<__nv_fp8_e5m2, double>(
            const __nv_fp8_e5m2 x)
    {
        return double(x);
    }
    #endif


    // ====
    // cast (double to __nv_fp8_e5m2)
    // ====

    /// \brief     Cast \c double type to \c __nv_fp8_e5m2 type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e5m2 cast<double, __nv_fp8_e5m2>(
            const double x)
    {
        return __nv_fp8_e5m2(x);
    }
    #endif


    // ====
    // cast (__nv_fp8_e4m3 to float)
    // ====

    /// \brief     Cast \c __nv_fp8_e4m3 type to \c float type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    inline __host__ __device__ float cast<__nv_fp8_e4m3, float>(
            const __nv_fp8_e4m3 x)
    {
        return float(x);
    }
    #endif


    // ====
    // cast (float to __nv_fp8_e4m3)
    // ====

    /// \brief     Cast \c float type to \c __nv_fp8_e4m3 type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e4m3 cast<float, __nv_fp8_e4m3>(
            const float x)
    {
        return __nv_fp8_e4m3(x);
    }
    #endif


    // ====
    // cast (__nv_fp8_e4m3 to double)
    // ====

    /// \brief     Cast \c __nv_fp8_e4m3 type to \c double type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    inline __host__ __device__ double cast<__nv_fp8_e4m3, double>(
            const __nv_fp8_e4m3 x)
    {
        return double(x);
    }
    #endif


    // ====
    // cast (double to __nv_fp8_e4m3)
    // ====

    /// \brief     Cast \c double type to \c __nv_fp8_e4m3 type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e4m3 cast<double, __nv_fp8_e4m3>(
            const double x)
    {
        return __nv_fp8_e4m3(x);
    }
    #endif


    // ====
    // cast (__half to float)
    // ====

    /// \brief     Cast \c __half type to \c float type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    inline __host__ __device__ float cast<__half, float>(const __half x)
    {
        return __half2float(x);
    }
    #endif


    // ====
    // cast (float to __half)
    // ====

    /// \brief     Cast \c float type to \c __half type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    inline __host__ __device__ __half cast<float, __half>(const float x)
    {
        return __float2half(x);
    }
    #endif


    // ====
    // cast (__half to double)
    // ====

    /// \brief     Cast \c __half type to \c double type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    inline __host__ __device__ double cast<__half, double>(const __half x)
    {
        return static_cast<double>(__half2float(x));
    }
    #endif


    // ====
    // cast (double to __half)
    // ====

    /// \brief     Cast \c double type to \c __half type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    inline __host__ __device__ __half cast<double, __half>(const double x)
    {
        return __float2half(static_cast<float>(x));
    }
    #endif


    // ====
    // cast (__nv_bfloat16 to float)
    // ====

    /// \brief     Cast \c __nv_bfloat16 type to \c float type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    inline __host__ __device__ float cast<__nv_bfloat16, float>(
            const __nv_bfloat16 x)
    {
        return __bfloat162float(x);
    }
    #endif


    // ====
    // cast (float to __nv_bfloat16)
    // ====

    /// \brief     Cast \c float type to \c __nv_bfloat16 type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    inline __host__ __device__ __nv_bfloat16 cast<float, __nv_bfloat16>(
            const float x)
    {
        return __float2bfloat16(x);
    }
    #endif


    // ====
    // cast (__nv_bfloat16 to double)
    // ====

    /// \brief     Cast \c __nv_bfloat16 type to \c double type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    inline __host__ __device__ double cast<__nv_bfloat16, double>(
            const __nv_bfloat16 x)
    {
        return static_cast<double>(__bfloat162float(x));
    }
    #endif


    // ====
    // cast (double to __nv_bfloat16)
    // ====

    /// \brief     Cast \c double type to \c __nv_bfloat16 type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    inline __host__ __device__ __nv_bfloat16 cast<double, __nv_bfloat16>(
            const double x)
    {
        return __float2bfloat16(static_cast<float>(x));
    }
    #endif


    // ====
    // cast (float to float)
    // ====

    /// \brief     Cast \c float type to \c float type (no action needed)
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ float cast<float, float>(
            const float x)
    {
        return x;
    }


    // ====
    // cast (float to double)
    // ====

    /// \brief     Cast \c float type to \c double type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ double cast<float, double>(
            const float x)
    {
        return static_cast<double>(x);
    }


    // ====
    // cast (double to double)
    // ====

    /// \brief     Cast \c double type to \c double type (no action needed)
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ double cast<double, double>(
            const double x)
    {
        return x;
    }


    // ====
    // cast (double to float)
    // ====

    /// \brief     Cast \c double type to \c float type.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ float cast<double, float>(
            const double x)
    {
        return static_cast<float>(x);
    }


    // ====
    // cast (int to __nv_fp8_e5m2)
    // ====

    /// \brief     Cast \c int type to \c __nv_fp8_e5m2 type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e5m2 cast<int, __nv_fp8_e5m2>(
            const int x)
    {
        return __nv_fp8_e5m2(x);
    }
    #endif


    // ====
    // cast (int to __nv_fp8_e4m3)
    // ====

    /// \brief     Cast \c int type to \c __nv_fp8_e4m3 type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e4m3 cast<int, __nv_fp8_e4m3>(
            const int x)
    {
        return __nv_fp8_e4m3(x);
    }
    #endif


    // ====
    // cast (int to __half)
    // ====

    /// \brief     Cast \c int type to \c __half type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    inline __host__ __device__ __half cast<int, __half>(const int x)
    {
        return __int2half_rn(x);
    }
    #endif


    // ====
    // cast (int to __nv_bfloat16)
    // ====

    /// \brief     Cast \c int type to \c __nv_bfloat16 type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    inline __host__ __device__ __nv_bfloat16 cast<int, __nv_bfloat16>(
            const int x)
    {
        return __int2bfloat16_rn(x);
    }
    #endif


    // ====
    // cast (int to float)
    // ====

    /// \brief     Cast \c int type to \c float type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ float cast<int, float>(
            const int x)
    {
        return static_cast<float>(x);
    }


    // ====
    // cast (float to int)
    // ====

    /// \brief     Cast \c float type to \c int type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ int cast<float, int>(
            const float x)
    {
        return static_cast<int>(x);
    }


    // ====
    // cast (int to double)
    // ====

    /// \brief     Cast \c int type to \c double type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ double cast<int, double>(
            const int x)
    {
        return static_cast<double>(x);
    }


    // ====
    // cast (double to int)
    // ====

    /// \brief     Cast \c double type to \c int type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ int cast<double, int>(
            const double x)
    {
        return static_cast<int>(x);
    }


    // ====
    // cast (unsigned int to __nv_fp8_e5m2)
    // ====

    /// \brief     Cast \c unsigned \c int type to \c __nv_fp8_e5m2 type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e5m2 cast<unsigned int, __nv_fp8_e5m2>(
            const unsigned int x)
    {
        return __nv_fp8_e5m2(x);
    }
    #endif


    // ====
    // cast (unsigned int to __nv_fp8_e4m3)
    // ====

    /// \brief     Cast \c unsigned \c int type to \c __nv_fp8_e4m3 type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e4m3 cast<unsigned int, __nv_fp8_e4m3>(
            const unsigned int x)
    {
        return __nv_fp8_e4m3(x);
    }
    #endif


    // ====
    // cast (unsigned int to __half)
    // ====

    /// \brief     Cast \c unsigned \c int type to \c __half type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    inline __host__ __device__ __half cast<unsigned int, __half>(
            const unsigned int x)
    {
        return __uint2half_rn(x);
    }
    #endif


    // ====
    // cast (unsigned int to __nv_bfloat16)
    // ====

    /// \brief     Cast \c unsigned \n int type to \c __nv_bfloat16 type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    inline __host__ __device__ __nv_bfloat16 cast<unsigned int, __nv_bfloat16>(
            const unsigned int x)
    {
        return __uint2bfloat16_rn(x);
    }
    #endif


    // ====
    // cast (unsigned int to float)
    // ====

    /// \brief     Cast \c unsigned \c int type to \c float type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ float cast<unsigned int, float>(
            const unsigned int x)
    {
        return static_cast<float>(x);
    }


    // ====
    // cast (float to unsigned int)
    // ====

    /// \brief     Cast \c float type to \c unsigned \c int type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ unsigned int cast<float, unsigned int>(
            const float x)
    {
        return static_cast<unsigned int>(x);
    }


    // ====
    // cast (unsigned int to double)
    // ====

    /// \brief     Cast \c unsigned \c int type to \c double type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ double cast<unsigned int, double>(
            const unsigned int x)
    {
        return static_cast<double>(x);
    }


    // ====
    // cast (double to unsigned int)
    // ====

    /// \brief     Cast \c double type to \c unsigned \c int type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ unsigned int cast<double, unsigned int>(
            const double x)
    {
        return static_cast<unsigned int>(x);
    }


    // ====
    // cast (long long int to __nv_fp8_e5m2)
    // ====

    /// \brief     Cast \c long \c long \c int type to \c __nv_fp8_e5m2 type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e5m2 cast<
        long long int, __nv_fp8_e5m2>(
            const long long int x)
    {
        return __nv_fp8_e5m2(x);
    }
    #endif


    // ====
    // cast (long long int to __nv_fp8_e4m3)
    // ====

    /// \brief     Cast \c long \c long \c int type to \c __nv_fp8_e4m3 type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e4m3 cast<
        long long int, __nv_fp8_e4m3>(
            const long long int x)
    {
        return __nv_fp8_e4m3(x);
    }
    #endif


    // ====
    // cast (long long int to __half)
    // ====

    /// \brief     Cast \c long \c long \c int type to \c __half type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    inline __host__ __device__ __half cast<long long int, __half>(
            const long long int x)
    {
        return __ll2half_rn(x);
    }
    #endif


    // ====
    // cast (long long int to __nv_bfloat16)
    // ====

    /// \brief     Cast \c long \c long \c int type to \c __nv_bfloat16 type
    ///            in round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    inline __host__ __device__ __nv_bfloat16 cast<
        long long int, __nv_bfloat16>(
            const long long int x)
    {
        return __ll2bfloat16_rn(x);
    }
    #endif


    // ====
    // cast (long long int to float)
    // ====

    /// \brief     Cast \c long \c long \c int type to \c float type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ float cast<long long int, float>(
            const long long int x)
    {
        return static_cast<float>(x);
    }


    // ====
    // cast (float to long long int)
    // ====

    /// \brief     Cast \c float type to \c long \c long \c int type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ long long int cast<float, long long int>(
            const float x)
    {
        return static_cast<long long int>(x);
    }


    // ====
    // cast (long long int to double)
    // ====

    /// \brief     Cast \c long \c long \c int type to \c double type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ double cast<long long int, double>(
            const long long int x)
    {
        return static_cast<double>(x);
    }


    // ====
    // cast (double to long long int)
    // ====

    /// \brief     Cast \c double type to \c long \c long \c int type in
    ///            round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ long long int cast<double, long long int>(
            const double x)
    {
        return static_cast<long long int>(x);
    }


    // ====
    // cast (unsigned long long int to __nv_fp8_e5m2)
    // ====

    /// \brief     Cast \c unsigned \c long \c long \c int type to
    ///            \c __nv_fp8_e5m2 type in round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e5m2 cast<
        unsigned long long int, __nv_fp8_e5m2>(
            const unsigned long long int x)
    {
        return __nv_fp8_e5m2(x);
    }
    #endif


    // ====
    // cast (unsigned long long int to __nv_fp8_e4m3)
    // ====

    /// \brief     Cast \c unsigned \c long \c long \c int type to
    ///            \c __nv_fp8_e4m3 type in round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    inline __host__ __device__ __nv_fp8_e4m3 cast<
        unsigned long long int, __nv_fp8_e4m3>(
            const unsigned long long int x)
    {
        return __nv_fp8_e4m3(x);
    }
    #endif


    // ====
    // cast (unsigned long long int to __half)
    // ====

    /// \brief     Cast \c unsigned \c long \c long \c int type to \c __half
    ///            type in round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    inline __host__ __device__ __half cast<unsigned long long int, __half>(
            const unsigned long long int x)
    {
        return __ull2half_rn(x);
    }
    #endif


    // ====
    // cast (unsigned long long int to __nv_bfloat16)
    // ====

    /// \brief     Cast \c unsigned \n long \c long \c int type to
    ///            \c __nv_bfloat16 type in round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    inline __host__ __device__ __nv_bfloat16 cast<
        unsigned long long int, __nv_bfloat16>(
            const unsigned long long int x)
    {
        return __ull2bfloat16_rn(x);
    }
    #endif


    // ====
    // cast (unsigned long long int to float)
    // ====

    /// \brief     Cast \c unsigned \c long \c long \c int type to \c float
    ///            type in round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ float cast<unsigned long long int, float>(
            const unsigned long long int x)
    {
        return static_cast<float>(x);
    }


    // ====
    // cast (float to unsigned long long int)
    // ====

    /// \brief     Cast \c float type to \c unsigned \c long \c long \c int
    ///            type in round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ unsigned long long int cast<
        float, unsigned long long int>(
            const float x)
    {
        return static_cast<unsigned long long int>(x);
    }


    // ====
    // cast (unsigned long long int to double)
    // ====

    /// \brief     Cast \c unsigned \c long \c long \c int type to \c double
    ///            type in round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ double cast<unsigned long long int, double>(
            const unsigned long long int x)
    {
        return static_cast<double>(x);
    }


    // ====
    // cast (double to unsigned long long int)
    // ====

    /// \brief     Cast \c double type to \c unsigned \c long \c long \c int
    ///            type in round-to-nearest-even mode.
    ///
    /// \param[in] x
    ///            Input
    /// \return    y
    ///            Output
    ///
    /// \sa        cu_arithmetic::add,
    ///            cu_arithmetic::mul
    
    template<>
    inline __host__ __device__ unsigned long long int cast<
        double, unsigned long long int>(
            const double x)
    {
        return static_cast<unsigned long long int>(x);
    }

}  // namespace cu_arithmetics

#endif  // _CU_ARITHMETICS_CU_CAST_H_
