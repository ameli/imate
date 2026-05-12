/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_BASIC_ALGEBRA_CBLAS_API_H_
#define _C_BASIC_ALGEBRA_CBLAS_API_H_


// =======
// Headers
// =======

#include "../_definitions/definitions.h"  // USE_CBLAS, USE_MKL, USE_ANY_CBLAS

#if defined(USE_ANY_CBLAS) && (USE_ANY_CBLAS == 1)

#if defined(USE_CBLAS) && (USE_CBLAS == 1)
    // Using OpenBLAS or similar
    #include <cblas.h>  // CBLAS_LAYOUT, CBLAS_TRANSPOSE, cblas_sgemv,
                        // cblas_dgemv, cblas_ssymv, cblas_dsymv, cblas_scopy,
                        // cblas_dcopy, cblas_saxpy, cblas_daxpy, cblas_snrm2,
                        // cblas_dnrm2, cblas_sscal, cblas_dscal
#elif defined(USE_MKL) && (USE_MKL == 1)
    // Using MKL
    #include <mkl_cblas.h>  // CBLAS_LAYOUT, CBLAS_TRANSPOSE, cblas_sgemv,
                            // cblas_dgemv, cblas_ssymv, cblas_dsymv,
                            // cblas_scopy, cblas_dcopy, cblas_saxpy,
                            // cblas_daxpy, cblas_snrm2, cblas_dnrm2,
                            // cblas_sscal, cblas_dscal
#endif

typedef CBLAS_ORDER CBLAS_LAYOUT;  // backward compatibility with CBLAS_LAYOUT

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


// ===============
// cblas interface
// ===============

/// \namespace cblas_api
///
/// \brief     A collection of templates to wrapper cblas functions.

namespace cblas_api
{
    // cblas xgemv
    template <typename DataType>
    void xgemv(
            const CBLAS_LAYOUT layout,
            const CBLAS_TRANSPOSE TransA,
            const int M,
            const int N,
            const DataType alpha,
            const DataType* RESTRICT A,
            const int lda,
            const DataType* RESTRICT X,
            const int incX,
            const DataType beta,
            DataType* RESTRICT Y,
            const int incY);

    // cblas xsymv
    template <typename DataType>
    void xsymv(
            const CBLAS_LAYOUT layout,
            const CBLAS_UPLO Uplo,
            const int N,
            const DataType alpha,
            const DataType* RESTRICT A,
            const int lda,
            const DataType* RESTRICT X,
            const int incX,
            const DataType beta,
            DataType* RESTRICT Y,
            const int incY);

    // cblas xcopy
    template <typename DataType>
    void xcopy(
            const int N,
            const DataType* RESTRICT X,
            const int incX,
            DataType* RESTRICT Y,
            const int incY);

    // cblas xaxpy
    template <typename DataType>
    void xaxpy(
            const int N,
            const DataType alpha,
            const DataType* RESTRICT X,
            const int incX,
            DataType* RESTRICT Y,
            const int incY);

    // cblas xdot
    template <typename DataType>
    DataType xdot(
            const int N,
            const DataType* RESTRICT X,
            const int incX,
            const DataType* RESTRICT Y,
            const int incY);

    // cblas xnrm2
    template <typename DataType>
    DataType xnrm2(
            const int N,
            const DataType* RESTRICT X,
            const int incX);

    // cblas xscal
    template <typename DataType>
    void xscal(
            const int N,
            const DataType alpha,
            DataType* RESTRICT X,
            const int incX);

}  // namespace cblas_api

#endif  // USE_ANY_CBLAS
#endif  // _C_BASIC_ALGEBRA_CBLAS_API_H_
