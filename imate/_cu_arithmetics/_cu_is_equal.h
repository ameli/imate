/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _CU_ARITHMETICS_CU_IS_EQUAL_H_
#define _CU_ARITHMETICS_CU_IS_EQUAL_H_


// =======
// Headers
// =======

#include <cmath>  // std::fabs
#include <limits>  // epsilon
#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __nv_bfloat16, __heq
#include <cassert>  // assert


// ==============
// cu arithmetics
// ==============

/// \namespace cu_arithmetics
///
/// \brief     This namespace declares arithmetic functions in CUDA.
///
/// \sa        cu_arithmetics::add,
///            cu_arithmetics::mul,
///            cu_arithmetics::cast

namespace cu_arithmetics
{

    // ========
    // is equal
    // ========

    /// \brief     Check if two floating point numbers are equal within a
    ///            tolerance.
    ///
    /// \param[in] x
    ///            A floating point number
    /// \param[in] y
    ///            A floating point number
    /// \return    \c true or \c false .

    template <typename DataType>
    inline bool is_equal(DataType x, DataType y);


    // ========
    // is equal (__nv_fp8_e5m2)
    // ========

    /// \brief     Check if two floating point numbers are equal within a
    ///            tolerance for \c __nv_fp8_e5m2 type.
    ///
    /// \param[in] x
    ///            A floating point number
    /// \param[in] y
    ///            A floating point number
    /// \return    \c true or \c false .

    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    inline bool is_equal(__nv_fp8_e5m2 x, __nv_fp8_e5m2 y)
    {
        // Not implemented
        assert(false);

        return false;
    }
    #endif


    // ========
    // is equal (__nv_fp8_e4m3)
    // ========

    /// \brief     Check if two floating point numbers are equal within a
    ///            tolerance for \c __nv_fp8_e4m3 type.
    ///
    /// \param[in] x
    ///            A floating point number
    /// \param[in] y
    ///            A floating point number
    /// \return    \c true or \c false .

    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    inline bool is_equal(__nv_fp8_e4m3 x, __nv_fp8_e4m3 y)
    {
        // Not implemented
        assert(false);

        return false;
    }
    #endif


    // ========
    // is equal (__half)
    // ========

    /// \brief     Check if two floating point numbers are equal within a
    ///            tolerance for \c __half type.
    ///
    /// \param[in] x
    ///            A floating point number
    /// \param[in] y
    ///            A floating point number
    /// \return    \c true or \c false .

    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    inline bool is_equal(__half x, __half y)
    {
        return __heq(x, y);
    }
    #endif


    // ========
    // is equal (__nv_bfloat16)
    // ========

    /// \brief     Check if two floating point numbers are equal within a
    ///            tolerance for \c __nv_bfloat16 type.
    ///
    /// \param[in] x
    ///            A floating point number
    /// \param[in] y
    ///            A floating point number
    /// \return    \c true or \c false .

    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    inline bool is_equal(__nv_bfloat16 x, __nv_bfloat16 y)
    {
        return __heq(x, y);
    }
    #endif


    // ========
    // is equal (float)
    // ========

    /// \brief     Check if two floating point numbers are equal within a
    ///            tolerance for \c float type.
    ///
    /// \param[in] x
    ///            A floating point number
    /// \param[in] y
    ///            A floating point number
    /// \return    \c true or \c false .

    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    inline bool is_equal(float x, float y)
    {
        if (std::fabs(x - y) <= 2.0 * std::numeric_limits<float>::epsilon())
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    #endif


    // ========
    // is equal (double)
    // ========

    /// \brief     Check if two floating point numbers are equal within a
    ///            tolerance for \c double type.
    ///
    /// \param[in] x
    ///            A floating point number
    /// \param[in] y
    ///            A floating point number
    /// \return    \c true or \c false .

    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    inline bool is_equal(double x, double y)
    {
        if (std::fabs(x - y) <= 2.0 * std::numeric_limits<double>::epsilon())
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    #endif

}  // namespace cu_arithmetics

#endif  // _CU_ARITHMETICS_CU_IS_EQUAL_H_
