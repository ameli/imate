/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _CU_BASIC_ALGEBRA_CUBLAS_IMPL_KERNELS_H_
#define _CU_BASIC_ALGEBRA_CUBLAS_IMPL_KERNELS_H_

// =======
// Headers
// =======

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


// ==================
// cublas impl kernel
// ==================

/// \namespace cublas_impl_kernels
///
/// \brief     Templated kernel code for implenentations of several BLAS-type
///            functions in CUDA.
///
/// \details   The motivation for re-implementing CuBLAS is that CUDA's CuBLAS
///            library does not supports \c DataType type and \c __nv_bfloat16
///            type for some of it functions. For instance, while there is
///            support for level 3 functions, they do not provide level 2 and
///            1 functions with \c DataType type.
///
///            The functions in this namespace provides some level 2 functions
///            by implementing CUDA kernels from scratch. These implementations
///            are templated with mixed precision computations where both the
///            \e data types and inner \e computation types are templated. The
///            data type is set by \c DatatType typename and the inner
///            computation type is set by \c ComputeType typename.
///
///            Despite the generic templated functions, he main intent of these
///            templates are to be used primarily for the missing types in
///            CuBLAS, namely, the \c DataType type (which is float16 type) and
///            \c __nv_bfloat6 type (which is Google's bfloat16 type). But
///            users may utilize these templates for any data and compute
///            types.
///
///            The prefix convension for all functions in this namespace are
///            \c cublasT (for instance \c cublasTgemv) where \c T here denotes
///            \e template. In the CuBLAS API, this letter a placeholder for
///            data type, such as \c S for single preicsion and \c D for double
///            precision.
///
///            The functions in this namespace are the \e host codes. The
///            \e kernel codes corresponding to each host code can be found
///            in \link cublas_impl_kernels \endlink namespace.
///
/// \sa        Namespace \link cublas_api cublas_api \endlink.

namespace cublas_impl_kernels
{
    // cublasTgemv kernel
    template <
        typename DataType, typename ComputeType, unsigned int block_size>
    __global__ void cublasTgemv_kernel(
            const bool trans,
            const int m,
            const int n,
            const DataType alpha,
            const DataType* RESTRICT A,
            const int lda,
            const DataType* RESTRICT x,
            const int incx,
            const DataType beta,
            DataType* RESTRICT y,
            const int incy);

    // cublasTcopy kernel
    template <typename DataType>
    __global__ void cublasTcopy_kernel(
            const int n,
            const DataType* RESTRICT x,
            const int incx,
            DataType* RESTRICT y,
            const int incy);

    // cublasTaxpy kernel
    template <typename DataType>
    __global__ void cublasTaxpy_kernel(
            const int n,
            const DataType alpha,
            const DataType* RESTRICT x,
            const int incx,
            DataType* RESTRICT y,
            const int incy);

    // cublasTdot kernel
    template <
        typename DataType, typename ComputeType, unsigned int block_size>
    __global__ void cublasTdot_kernel(
            const int n,
            const DataType* RESTRICT x,
            const int incx,
            const DataType* RESTRICT y,
            const int incy,
            ComputeType* RESTRICT result);

    // cublasTnrm2 kernel
    template <
        typename DataType, typename ComputeType, unsigned int block_size>
    __global__ void cublasTnrm2_kernel(
            const int n,
            const DataType* RESTRICT x,
            const int incx,
            ComputeType* RESTRICT result);

    // cublasTscal kernel
    template <typename DataType>
    __global__ void cublasTscal_kernel(
            const int n,
            const DataType alpha,
            DataType* RESTRICT x,
            const int incx);
}

#endif  // _CU_BASIC_ALGEBRA_CUBLAS_IMPL_KERNELS_H_
