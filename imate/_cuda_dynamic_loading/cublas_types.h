/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CUDA_DYNAMIC_LOADING_CUBLAS_TYPES_H_
#define _CUDA_DYNAMIC_LOADING_CUBLAS_TYPES_H_

// =======
// Headers
// =======

#include <cublas_v2.h>  // cublasSgemv, cublasDgemv, cublasScopy, cublasDcopy,
                        // cublasSaxpy, cublasDaxpy, cublasSdot, cublasDdot,
                        // cublasSnrm2, cublasDnrm2, cublasSscal, cublasDscal
                        // cublasHandle_t, cublasStatus_t

// =====
// Types
// =====

// cublasCreate
typedef cublasStatus_t (*cublasCreate_type)(cublasHandle_t* handle);

// cublasDestroy
typedef cublasStatus_t (*cublasDestroy_type)(cublasHandle_t handle);

// cublasSgemv
typedef cublasStatus_t (*cublasSgemv_type)(
        cublasHandle_t handle,
        cublasOperation_t trans,
        int m,
        int n,
        const float* alpha,
        const float* A,
        int lda,
        const float* x,
        int incx,
        const float* beta,
        float* y,
        int incy);

// cublasDgemv
typedef cublasStatus_t (*cublasDgemv_type)(
        cublasHandle_t handle,
        cublasOperation_t trans,
        int m,
        int n,
        const double* alpha,
        const double* A,
        int lda,
        const double* x,
        int incx,
        const double* beta,
        double* y,
        int incy);

// cublasScopy
typedef cublasStatus_t (*cublasScopy_type)(
        cublasHandle_t handle, int n,
        const float* x,
        int incx,
        float* y,
        int incy);

// cublasDcopy
typedef cublasStatus_t (*cublasDcopy_type)(
        cublasHandle_t handle, int n,
        const double* x,
        int incx,
        double* y,
        int incy);

// cublasSaxpy
typedef cublasStatus_t (*cublasSaxpy_type)(
        cublasHandle_t handle,
        int n,
        const float *alpha,
        const float *x,
        int incx,
        float *y,
        int incy);

// cublasDaxpy
typedef cublasStatus_t (*cublasDaxpy_type)(
        cublasHandle_t handle,
        int n,
        const double *alpha,
        const double *x,
        int incx,
        double *y,
        int incy);

// cublasSdot
typedef cublasStatus_t (*cublasSdot_type)(
        cublasHandle_t handle,
        int n,
        const float *x,
        int incx,
        const float *y,
        int incy,
        float *result);

// cublasDdot
typedef cublasStatus_t (*cublasDdot_type)(
        cublasHandle_t handle,
        int n,
        const double *x,
        int incx,
        const double *y,
        int incy,
        double *result);

// cublasSnrm2
typedef cublasStatus_t (*cublasSnrm2_type)(
        cublasHandle_t handle,
        int n,
        const float *x,
        int incx,
        float *result);

// cublasDnrm2
typedef cublasStatus_t (*cublasDnrm2_type)(
        cublasHandle_t handle,
        int n,
        const double *x,
        int incx,
        double *result);

// cublasSscal
typedef cublasStatus_t (*cublasSscal_type)(
        cublasHandle_t handle,
        int n,
        const float *alpha,
        float *x,
        int incx);

// cublasDscal
typedef cublasStatus_t (*cublasDscal_type)(
        cublasHandle_t handle,
        int n,
        const double *alpha,
        double *x,
        int incx);

#endif  // _CUDA_DYNAMIC_LOADING_CUBLAS_TYPES_H_
