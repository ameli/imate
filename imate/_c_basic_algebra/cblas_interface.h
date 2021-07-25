/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_BASIC_ALGEBRA_CBLAS_INTERFACE_H_
#define _C_BASIC_ALGEBRA_CBLAS_INTERFACE_H_


// =======
// Headers
// =======

#include "../_definitions/definitions.h"  // USE_CBLAS

#if (USE_CBLAS == 1)

#include <cblas.h>  // CBLAS_LAYOUT, CBLAS_TRANSPOSE, cblas_sgemv, cblas_dgemv,
                    // cblas_scopy, cblas_dcopy, cblas_saxpy, cblas_daxpy,
                    // cblas_snrm2, cblas_dnrm2, cblas_sscal, cblas_dscal

typedef CBLAS_ORDER CBLAS_LAYOUT;  // backward compatibility with CBLAS_LAYOUT


// ===============
// cblas interface
// ===============

/// \namespace cblas_interface
///
/// \brief     A collection of templates to wrapper cblas functions.

namespace cblas_interface
{
    // cblas xgemv
    template <typename DataType>
    void xgemv(
            CBLAS_LAYOUT layout,
            CBLAS_TRANSPOSE TransA,
            const int M,
            const int N,
            const DataType alpha,
            const DataType* A,
            const int lda,
            const DataType* X,
            const int incX,
            const DataType beta,
            DataType* Y,
            const int incY);

    // cblas xcopy
    template <typename DataType>
    void xcopy(
            const int N,
            const DataType* X,
            const int incX,
            DataType* Y,
            const int incY);

    // cblas xaxpy
    template <typename DataType>
    void xaxpy(
            const int N,
            const DataType alpha,
            const DataType* X,
            const int incX,
            DataType* Y,
            const int incY);

    // cblas xdot
    template <typename DataType>
    DataType xdot(
            const int N,
            const DataType* X,
            const int incX,
            const DataType* Y,
            const int incY);

    // cblas xnrm2
    template <typename DataType>
    DataType xnrm2(
            const int N,
            const DataType* X,
            const int incX);

    // cblas xscal
    template <typename DataType>
    void xscal(
            const int N,
            const DataType alpha,
            DataType* X,
            const int incX);

}  // namespace cblas_interface

#endif  // USE_CBLAS
#endif  // _C_BASIC_ALGEBRA_CBLAS_INTERFACE_H_
