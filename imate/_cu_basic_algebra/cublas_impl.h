/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _CU_BASIC_ALGEBRA_CUBLAS_IMPL_H_
#define _CU_BASIC_ALGEBRA_CUBLAS_IMPL_H_

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
    #include <cublas_v2.h>  // cublasOperation_t, CUBLAS_OP_N, CUBLAS_OP_T,
                            // cudaError_t
#endif

// Restrict qualifier
// Note: generally, the restrict qualifier is only needed to be be added to the
// function signature in the implementation (*.cu files) and not the
// declarations (in *h files). However, CUDA seems to have a bug that the
// restrict qualifier should be added to BOTH declaration and implementation,
// otherwise, it corrupts the array pointers.
#if defined(_MSC_VER)
    #define RESTRICT __restrict
#elif defined(__INTEL_COMPILER)
    #define RESTRICT __restrict
#elif defined(__CUDA__) || defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#else
    #define RESTRICT
#endif


// ===========
// cublas impl
// ===========

/// \namespace cublas_impl
///
/// \brief     Templated implenentations of several BLAS-type functions in
///            CUDA.
///
/// \details   The motivation for re-implementing CuBLAS is that CUDA's CuBLAS
///            library does not supports \c __half type and \c __nv_bfloat16
///            type for some of it functions. For instance, while there is
///            support for level 3 functions, they do not provide level 2 and
///            1 functions with \c __half type.
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
///            CuBLAS, namely, the \c __half type (which is float16 type) and
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

namespace cublas_impl
{
    // cublasTgemv 
    template <typename DataType, typename ComputeType>
    cudaError_t cublasTgemv(
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

    // cublasTcopy
    template <typename DataType>
    cudaError_t cublasTcopy(
            int n,
            const DataType* RESTRICT x,
            int incx,
            DataType* RESTRICT y,
            int incy);

    // cublasTaxpy
    template <typename DataType>
    cudaError_t cublasTaxpy(
            int n,
            const DataType* RESTRICT alpha,
            const DataType* RESTRICT x,
            int incx,
            DataType* RESTRICT y,
            int incy);

    // cublasTdot
    template <typename DataType, typename ComputeType>
    cudaError_t cublasTdot(
            int n,
            const DataType* RESTRICT x,
            int incx,
            const DataType* RESTRICT y,
            int incy,
            DataType* RESTRICT result);

    // cublasTnrm2
    template <typename DataType, typename ComputeType>
    cudaError_t cublasTnrm2(
            int n,
            const DataType* RESTRICT x,
            int incx,
            DataType* RESTRICT result);

    // cublasTscal
    template <typename DataType>
    cudaError_t cublasTscal(
            int n,
            const DataType* RESTRICT alpha,
            DataType* RESTRICT x,
            int incx);
}

#endif  // _CU_BASIC_ALGEBRA_CUBLAS_IMPL_H_
