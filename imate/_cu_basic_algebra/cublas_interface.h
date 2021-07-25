/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _CU_BASIC_ALGEBRA_CUBLAS_INTERFACE_H_
#define _CU_BASIC_ALGEBRA_CUBLAS_INTERFACE_H_


// =======
// Headers
// =======

#include <cublas_v2.h>  // cublasSgemv, cublasDgemv, cublasScopy, cublasDcopy,
                        // cublasSaxpy, cublasDaxpy, cublasSdot, cublasDdot,
                        // cublasSnrm2, cublasDnrm2, cublasSscal, cublasDscal
                        // cublasHandle_t, cublasStatus_t


// ================
// cublas interface
// ================

/// \namespace cublas_interface
///
/// \brief     A collection of templates to wrapper cublas functions.

namespace cublas_interface
{
    // cublasXgemv
    template <typename DataType>
    cublasStatus_t cublasXgemv(
            cublasHandle_t handle,
            cublasOperation_t trans,
            int m,
            int n,
            const DataType* alpha,
            const DataType* A,
            int lda,
            const DataType* x,
            int incx,
            const DataType* beta,
            DataType* y,
            int incy);

    // cublasXcopy
    template <typename DataType>
    cublasStatus_t cublasXcopy(
            cublasHandle_t handle, int n,
            const DataType* x,
            int incx,
            DataType* y,
            int incy);

    // cublasXaxpy
    template <typename DataType>
    cublasStatus_t cublasXaxpy(
            cublasHandle_t handle,
            int n,
            const DataType *alpha,
            const DataType *x,
            int incx,
            DataType *y,
            int incy);

    // cublasXdot
    template <typename DataType>
    cublasStatus_t cublasXdot(
            cublasHandle_t handle,
            int n,
            const DataType *x,
            int incx,
            const DataType *y,
            int incy,
            DataType *result);

    // cublasXnrm2
    template <typename DataType>
    cublasStatus_t cublasXnrm2(
            cublasHandle_t handle,
            int n,
            const DataType *x,
            int incx,
            DataType *result);

    // cublasXscal
    template <typename DataType>
    cublasStatus_t cublasXscal(
            cublasHandle_t handle,
            int n,
            const DataType *alpha,
            DataType *x,
            int incx);
}  // namespace cublas_interface


#endif  //  _CU_BASIC_ALGEBRA_CUBLAS_INTERFACE_H_
