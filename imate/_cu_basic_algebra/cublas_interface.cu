/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


// =======
// Headers
// =======

#include "./cublas_interface.h"


// ================
// cublas interface
// ================

/// \note      The implementation in the \c cu file is wrapped inside the
///            namepsace clause. This is not necessary in general, however, it
///            is needed to avoid the old gcc compiler error (this is a gcc
///            bug) which complains "no instance of function template matches
///            the argument list const float".

namespace cublas_interface
{

    // ===========
    // cublasXgemv (float)
    // ===========

    /// \brief A template wrapper for \c cublasSgemv.
    ///

    template<>
    cublasStatus_t cublasXgemv<float>(
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
            int incy)
    {
        return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta,
                           y, incy);
    }


    // ===========
    // cublasXgemv (double)
    // ===========

    /// \brief A template wrapper for \c cublasDgemv.
    ///

    template<>
    cublasStatus_t cublasXgemv<double>(
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
            int incy)
    {
        return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta,
                           y, incy);
    }



    // ===========
    // cublasXcopy (float)
    // ===========

    /// \brief  A template wrapper for \c cublasScopy.
    ///

    template<>
    cublasStatus_t cublasXcopy<float>(
            cublasHandle_t handle,
            int n,
            const float* x,
            int incx,
            float* y,
            int incy)
    {
        return cublasScopy(handle, n, x, incx, y, incy);
    }


    // ===========
    // cublasXcopy (double)
    // ===========

    /// \brief  A template wrapper for \c cublasDcopy.
    ///

    template<>
    cublasStatus_t cublasXcopy<double>(
            cublasHandle_t handle,
            int n,
            const double* x,
            int incx,
            double* y,
            int incy)
    {
        return cublasDcopy(handle, n, x, incx, y, incy);
    }


    // ===========
    // cublasXaxpy (float)
    // ===========

    /// \brief A template wrapper for \c cublasSaxpy
    ///

    template<>
    cublasStatus_t cublasXaxpy<float>(
            cublasHandle_t handle,
            int n,
            const float *alpha,
            const float *x,
            int incx,
            float *y,
            int incy)
    {
        return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
    }


    // ===========
    // cublasXaxpy (double)
    // ===========

    /// \brief A template wrapper for \c cublasDaxpy
    ///

    template<>
    cublasStatus_t cublasXaxpy<double>(
            cublasHandle_t handle,
            int n,
            const double *alpha,
            const double *x,
            int incx,
            double *y,
            int incy)
    {
        return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
    }


    // ==========
    // cublasXdot (float)
    // ==========

    /// \brief A template wrapper for \c cublasSdot
    ///

    template<>
    cublasStatus_t cublasXdot<float>(
            cublasHandle_t handle,
            int n,
            const float *x,
            int incx,
            const float *y,
            int incy,
            float *result)
    {
        return cublasSdot(handle, n, x, incx, y, incy, result);
    }


    // ==========
    // cublasXdot (double)
    // ==========

    /// \brief A template wrapper for \c cublasDdot
    ///

    template<>
    cublasStatus_t cublasXdot<double>(
            cublasHandle_t handle,
            int n,
            const double *x,
            int incx,
            const double *y,
            int incy,
            double *result)
    {
        return cublasDdot(handle, n, x, incx, y, incy, result);
    }


    // ===========
    // cublasXnrm2 (float)
    // ===========

    /// \brief A template wrapper to \c cublasSnrm2
    ///

    template<>
    cublasStatus_t cublasXnrm2<float>(
            cublasHandle_t handle,
            int n,
            const float *x,
            int incx,
            float *result)
    {
        return cublasSnrm2(handle, n, x, incx, result);
    }


    // ===========
    // cublasXnrm2 (double)
    // ===========

    /// \brief A template wrapper to \c cublasDnrm2
    ///

    template<>
    cublasStatus_t cublasXnrm2<double>(
            cublasHandle_t handle,
            int n,
            const double *x,
            int incx,
            double *result)
    {
        return cublasDnrm2(handle, n, x, incx, result);
    }


    // ===========
    // cublasXscal (float)
    // ===========

    /// \brief A template wrapper for \c cublasSscal.
    ///

    template<>
    cublasStatus_t cublasXscal<float>(
            cublasHandle_t handle,
            int n,
            const float *alpha,
            float *x,
            int incx)
    {
        return cublasSscal(handle, n, alpha, x, incx);
    }


    // ===========
    // cublasXscal (double)
    // ===========

    /// \brief A template wrapper for \c cublasDscal.
    ///

    template<>
    cublasStatus_t cublasXscal<double>(
            cublasHandle_t handle,
            int n,
            const double *alpha,
            double *x,
            int incx)
    {
        return cublasDscal(handle, n, alpha, x, incx);
    }
}  // namespace cublas_interface
