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

#include "./cblas_api.h"

# if defined(USE_ANY_CBLAS) && (USE_ANY_CBLAS == 1)

#include <cstdlib>  // abort
#include <iostream>  // std::cerr


// ===============
// cblas interface
// ===============

/// \note      The implementation in the \c cpp file is wrapped inside the
///            namepsace clause. This is not necessary in general, however, it
///            is needed to avoid the old gcc compiler error (this is a gcc
///            bug) which complains "no instance of function template matches
///            the argument list const float".

namespace cblas_api
{

    // =====
    // xgemv (float)
    // =====

    /// \brief A template wrapper for \c cblas_sgemv.
    ///

    template<>
    void xgemv<float>(
            const CBLAS_LAYOUT layout,
            const CBLAS_TRANSPOSE TransA,
            const int M,
            const int N,
            const float alpha,
            const float* RESTRICT A,
            const int lda,
            const float* RESTRICT X,
            const int incX,
            const float beta,
            float* RESTRICT Y,
            const int incY)
    {
        cblas_sgemv(layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                    incY);
    }


    // =====
    // xgemv (double)
    // =====

    /// \brief A template wrapper for \c cblas_dgemv.
    ///

    template<>
    void xgemv<double>(
            const CBLAS_LAYOUT layout,
            const CBLAS_TRANSPOSE TransA,
            const int M,
            const int N,
            const double alpha,
            const double* RESTRICT A,
            const int lda,
            const double* RESTRICT X,
            const int incX,
            const double beta,
            double* RESTRICT Y,
            const int incY)
    {
        cblas_dgemv(layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                    incY);
    }


    // =====
    // xgemv (long double)
    // =====

    /// \brief This function is not implemented in CBLAS.
    ///

    template<>
    void xgemv<long double>(
            const CBLAS_LAYOUT layout,
            const CBLAS_TRANSPOSE TransA,
            const int M,
            const int N,
            const long double alpha,
            const long double* RESTRICT A,
            const int lda,
            const long double* RESTRICT X,
            const int incX,
            const long double beta,
            long double* RESTRICT Y,
            const int incY)
    {
        // Mark unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) layout;
        (void) TransA;
        (void) M;
        (void) N;
        (void) alpha;
        (void) A;
        (void) lda;
        (void) X;
        (void) incX;
        (void) beta;
        (void) Y;
        (void) incY;

        std::cerr << "Error: cblas_?gemv for long double type is not "
                  << "implemented. To use long double type, set USE_CBLAS "
                  << "and USE_MKL to 0 and recompile the package."
                  << std::endl;
        abort();
    }


    // =====
    // xsymv (float)
    // =====

    /// \brief A template wrapper for \c cblas_ssymv.
    ///

    template<>
    void xsymv<float>(
            const CBLAS_LAYOUT layout,
            const CBLAS_UPLO Uplo,
            const int N,
            const float alpha,
            const float* RESTRICT A,
            const int lda,
            const float* RESTRICT X,
            const int incX,
            const float beta,
            float* RESTRICT Y,
            const int incY)
    {
        cblas_ssymv(layout, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
    }


    // =====
    // xsymv (double)
    // =====

    /// \brief A template wrapper for \c cblas_dsymv.
    ///

    template<>
    void xsymv<double>(
            const CBLAS_LAYOUT layout,
            const CBLAS_UPLO Uplo,
            const int N,
            const double alpha,
            const double* RESTRICT A,
            const int lda,
            const double* RESTRICT X,
            const int incX,
            const double beta,
            double* RESTRICT Y,
            const int incY)
    {
        cblas_dsymv(layout, Uplo, N, alpha, A, lda, X, incX, beta, Y, incY);
    }


    // =====
    // xsymv (long double)
    // =====

    /// \brief This function is not implemented in CBLAS.
    ///

    template<>
    void xsymv<long double>(
            const CBLAS_LAYOUT layout,
            const CBLAS_UPLO Uplo,
            const int N,
            const long double alpha,
            const long double* RESTRICT A,
            const int lda,
            const long double* RESTRICT X,
            const int incX,
            const long double beta,
            long double* RESTRICT Y,
            const int incY)
    {
        // Mark unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) layout;
        (void) Uplo;
        (void) N;
        (void) alpha;
        (void) A;
        (void) lda;
        (void) X;
        (void) incX;
        (void) beta;
        (void) Y;
        (void) incY;

        std::cerr << "Error: cblas_?symv for long double type is not "
                  << "implemented. To use long double type, set USE_CBLAS "
                  << "and USE_MKL to 0 and recompile the package."
                  << std::endl;
        abort();
    }


    // =====
    // xcopy (float)
    // =====

    /// \brief A template wrapper for \c cblas_scopy.
    ///

    template <>
    void xcopy<float>(
            const int N,
            const float* RESTRICT X,
            const int incX,
            float* RESTRICT Y,
            const int incY)
    {
        cblas_scopy(N, X, incX, Y, incY);
    }


    // =====
    // xcopy (double)
    // =====

    /// \brief A template wrapper for \c cblas_dcopy.
    ///

    template <>
    void xcopy<double>(
            const int N,
            const double* RESTRICT X,
            const int incX,
            double* RESTRICT Y,
            const int incY)
    {
        cblas_dcopy(N, X, incX, Y, incY);
    }


    // =====
    // xcopy (long double)
    // =====

    /// \brief This function is not implemented in CBLAS.
    ///

    template <>
    void xcopy<long double>(
            const int N,
            const long double* RESTRICT X,
            const int incX,
            long double* RESTRICT Y,
            const int incY)
    {
        // Mark unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) N;
        (void) X;
        (void) incX;
        (void) Y;
        (void) incY;

        std::cerr << "Error: cblas_?copy for long double type is not "
                  << "implemented. To use long double type, set USE_CBLAS "
                  << "and USE_MKL to 0 and recompile the package."
                  << std::endl;
        abort();
    }


    // =====
    // xaxpy (float)
    // =====

    /// \brief A template wrapper for \c cblas_saxpy.
    ///

    template <>
    void xaxpy<float>(
            const int N,
            const float alpha,
            const float* RESTRICT X,
            const int incX,
            float* RESTRICT Y,
            const int incY)
    {
        cblas_saxpy(N, alpha, X, incX, Y, incY);
    }


    // =====
    // xaxpy (double)
    // =====

    /// \brief A template wrapper for \c cblas_daxpy.
    ///

    template <>
    void xaxpy<double>(
            const int N,
            const double alpha,
            const double* RESTRICT X,
            const int incX,
            double* RESTRICT Y,
            const int incY)
    {
        cblas_daxpy(N, alpha, X, incX, Y, incY);
    }


    // =====
    // xaxpy (long double)
    // =====

    /// \brief This function is not implemented in CBLAS.
    ///

    template <>
    void xaxpy<long double>(
            const int N,
            const long double alpha,
            const long double* RESTRICT X,
            const int incX,
            long double* RESTRICT Y,
            const int incY)
    {
        // Mark unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) N;
        (void) alpha;
        (void) X;
        (void) incX;
        (void) Y;
        (void) incY;

        std::cerr << "Error: cblas_?axpy for long double type is not "
                  << "implemented. To use long double type, set USE_CBLAS "
                  << "and USE_MKL to 0 and recompile the package."
                  << std::endl;
        abort();
    }


    // ====
    // xdot (float)
    // ====

    /// \brief A template wrapper for \c cblas_sdot.
    ///

    template <>
    float xdot<float>(
            const int N,
            const float* RESTRICT X,
            const int incX,
            const float* RESTRICT Y,
            const int incY)
    {
        return cblas_sdot(N, X, incX, Y, incY);
    }


    // ====
    // xdot (double)
    // ====

    /// \brief A template wrapper for \c cblas_ddot.
    ///

    template <>
    double xdot<double>(
            const int N,
            const double* RESTRICT X,
            const int incX,
            const double* RESTRICT Y,
            const int incY)
    {
        return cblas_ddot(N, X, incX, Y, incY);
    }


    // ====
    // xdot (long double)
    // ====

    /// \brief This function is not implemented in CBLAS.
    ///

    template <>
    long double xdot<long double>(
            const int N,
            const long double* RESTRICT X,
            const int incX,
            const long double* RESTRICT Y,
            const int incY)
    {
        // Mark unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) N;
        (void) X;
        (void) incX;
        (void) Y;
        (void) incY;

        std::cerr << "Error: cblas_?dot for long double type is not "
                  << "implemented. To use long double type, set USE_CBLAS "
                  << "and USE_MKL to 0 and recompile the package."
                  << std::endl;
        abort();
    }


    // =====
    // xnrm2 (float)
    // =====

    /// \brief A template wrapper for \c cblas_snrm2.
    ///

    template <>
    float xnrm2<float>(
            const int N,
            const float* RESTRICT X,
            const int incX)
    {
        return cblas_snrm2(N, X, incX);
    }


    // =====
    // xnrm2 (double)
    // =====

    /// \brief A template wrapper for \c cblas_dnrm2.
    ///

    template <>
    double xnrm2<double>(
            const int N,
            const double* RESTRICT X,
            const int incX)
    {
        return cblas_dnrm2(N, X, incX);
    }


    // =====
    // xnrm2 (long double)
    // =====

    /// \brief This function is not implemented in CBLAS.
    ///

    template <>
    long double xnrm2<long double>(
            const int N,
            const long double* RESTRICT X,
            const int incX)
    {
        // Mark unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) N;
        (void) X;
        (void) incX;

        std::cerr << "Error: cblas_?nrm2 for long double type is not "
                  << "implemented. To use long double type, set USE_CBLAS "
                  << "i and USE_MKL to 0 and recompile the package."
                  << std::endl;
        abort();
    }


    // =====
    // xscal (float)
    // =====

    /// \brief A template wrapper for \c cblas_sscal.
    ///

    template <>
    void xscal<float>(
            const int N,
            const float alpha,
            float* RESTRICT X,
            const int incX)
    {
        cblas_sscal(N, alpha, X, incX);
    }


    // =====
    // xscal (double)
    // =====

    /// \brief A template wrapper for \c cblas_dscal.
    ///

    template <>
    void xscal<double>(
            const int N,
            const double alpha,
            double* RESTRICT X,
            const int incX)
    {
        cblas_dscal(N, alpha, X, incX);
    }


    // =====
    // xscal (long double)
    // =====

    /// \brief This function is not implemented in CBLAS.
    ///

    template <>
    void xscal<long double>(
            const int N,
            const long double alpha,
            long double* RESTRICT X,
            const int incX)
    {
        // Mark unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) N;
        (void) alpha;
        (void) X;
        (void) incX;

        std::cerr << "Error: cblas_?scal for long double type is not "
                  << "implemented. To use long double type, set USE_CBLAS "
                  << "and USE_MKL to 0 and recompile the package."
                  << std::endl;
        abort();
    }
}  // namespace cblas_api

#endif  // USE_ANY_CBLAS
