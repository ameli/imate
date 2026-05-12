/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _CU_BASIC_ALGEBRA_CUBLAS_API_H_
#define _CU_BASIC_ALGEBRA_CUBLAS_API_H_


// =======
// Headers
// =======

// Avoid CUBLAS numeration value not handled in switch [-Wswitch-enum] warning
#ifdef _MSC_VER
    #pragma warning(push, 0)  // Suppress all warnings from the followings
    #include <cublas_v2.h>
    #pragma warning(pop)  // Restore previous warning level
#elif defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
    #pragma warning(push, 0)
    #include <cublas_v2.h>
    #pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wswitch-enum"
    #include <cublas_v2.h>
    #pragma GCC diagnostic pop
#else
    #include <cublas_v2.h>  // cublasSgemv, cublasDgemv, cublasScopy,
                            // cublasDcopy, cublasSaxpy, cublasDaxpy,
                            // cublasSdot, cublasDdot, cublasSnrm2,
                            // cublasDnrm2, cublasSscal, cublasDscal
                            // cublasHandle_t, cublasStatus_t
#endif

// Restrict qualifier
#if defined(_MSC_VER)
    #define RESTRICT __restrict
#elif defined(__INTEL_COMPILER)
    #define RESTRICT __restrict
#elif defined(__CUDA__) || defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#else
    #define RESTRICT
#endif


// ==========
// cublas api
// ==========

/// \namespace cublas_api
///
/// \brief     A collection of templates to wrapper cublas functions.

namespace cublas_api
{
    // cublasXgemv
    template <typename DataType>
    cublasStatus_t cublasXgemv(
            cublasHandle_t handle,
            cublasOperation_t trans,
            int m,
            int n,
            const DataType* RESTRICT alpha,
            const DataType* RESTRICT A,
            int lda,
            const DataType* RESTRICT x,
            int incx,
            const DataType* RESTRICT beta,
            DataType* RESTRICT y,
            int incy);

    // cublasXcopy
    template <typename DataType>
    cublasStatus_t cublasXcopy(
            cublasHandle_t handle,
            int n,
            const DataType* RESTRICT x,
            int incx,
            DataType* RESTRICT y,
            int incy);

    // cublasXaxpy
    template <typename DataType>
    cublasStatus_t cublasXaxpy(
            cublasHandle_t handle,
            int n,
            const DataType* RESTRICT alpha,
            const DataType* RESTRICT x,
            int incx,
            DataType* RESTRICT y,
            int incy);

    // cublasXdot
    template <typename DataType>
    cublasStatus_t cublasXdot(
            cublasHandle_t handle,
            int n,
            const DataType* RESTRICT x,
            int incx,
            const DataType* RESTRICT y,
            int incy,
            DataType* RESTRICT result);

    // cublasXnrm2
    template <typename DataType>
    cublasStatus_t cublasXnrm2(
            cublasHandle_t handle,
            int n,
            const DataType* RESTRICT x,
            int incx,
            DataType* RESTRICT result);

    // cublasXscal
    template <typename DataType>
    cublasStatus_t cublasXscal(
            cublasHandle_t handle,
            int n,
            const DataType* RESTRICT alpha,
            DataType* RESTRICT x,
            int incx);

}  // namespace cublas_api


#endif  //  _CU_BASIC_ALGEBRA_CUBLAS_API_H_
