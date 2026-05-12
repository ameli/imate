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

#include "./cublas_api.h"
#include <cuda_runtime.h>  // cudaError_t, cudaSuccess
#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __nv_bfloat16
#include "./cublas_impl.h"  // cublas_impl


// ==========
// cublas api
// ==========

/// \note      The implementation in the \c cu file is wrapped inside the
///            namepsace clause. This is not necessary in general, however, it
///            is needed to avoid the old gcc compiler error (this is a gcc
///            bug) which complains "no instance of function template matches
///            the argument list const float".

namespace cublas_api
{
    // ===========
    // cublasXgemv (__nv_fp8_e5m2)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \alpha \text{op}(\mathbf{A})
    ///             \boldsymbol{x} + \beta \boldsymbol{y} \f$.
    ///
    /// \details    This function is not a template wrapper for CuBLAS, since
    ///             CuBLAS API does not have \c cublasHgemv (where \c H is
    ///             used for \c __nv_fp8_e5m2 type). As such, this function is
    ///             implemented with CUDA, rather than from CuBLAS.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  trans
    ///             If set to \c CUBLAS_OP_N or \c CUBLAS_OP_T, the operator
    ///             \f$ \mathbf{A} \f$ is not transposed or transposed,
    ///             respectively.
    /// \param[in]  m
    ///             Number of rows of matrix \f$ \mathbf{A} \f$.
    /// \param[in]  n
    ///             Number of columns of matrix \f$ \mathbf{A} \f$.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$.
    /// \param[in]  A
    ///             Two-dimensional matrix \f$ \mathbf{A} \f$ stored on GPU
    ///             device as one-dimensional array with column-major ordering.
    /// \param[in]  lda
    ///             Leading dimension of two-dimensional matrix
    ///             \f$ \mathbf{A} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    ///
    /// \sa         cublasXaxpy

    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    cublasStatus_t cublasXgemv<__nv_fp8_e5m2>(
            cublasHandle_t handle,
            cublasOperation_t trans,
            int m,
            int n,
            const __nv_fp8_e5m2* RESTRICT alpha,
            const __nv_fp8_e5m2* RESTRICT A,
            int lda,
            const __nv_fp8_e5m2* RESTRICT x,
            int incx,
            const __nv_fp8_e5m2* RESTRICT beta,
            __nv_fp8_e5m2* RESTRICT y,
            int incy)
    {
        // Void unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) handle;

        cudaError_t error = cublas_impl::cublasTgemv<__nv_fp8_e5m2, float>(
                trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        
        if (error != cudaSuccess)
        {
            return CUBLAS_STATUS_SUCCESS;
        }
        else
        {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    #endif

    
    // ===========
    // cublasXgemv (__nv_fp8_e4m3)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \alpha \text{op}(\mathbf{A})
    ///             \boldsymbol{x} + \beta \boldsymbol{y} \f$.
    ///
    /// \details    This function is not a template wrapper for CuBLAS, since
    ///             CuBLAS API does not have \c cublasHgemv (where \c H is
    ///             used for \c __nv_fp8_e4m3 type). As such, this function is
    ///             implemented with CUDA, rather than from CuBLAS.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  trans
    ///             If set to \c CUBLAS_OP_N or \c CUBLAS_OP_T, the operator
    ///             \f$ \mathbf{A} \f$ is not transposed or transposed,
    ///             respectively.
    /// \param[in]  m
    ///             Number of rows of matrix \f$ \mathbf{A} \f$.
    /// \param[in]  n
    ///             Number of columns of matrix \f$ \mathbf{A} \f$.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$.
    /// \param[in]  A
    ///             Two-dimensional matrix \f$ \mathbf{A} \f$ stored on GPU
    ///             device as one-dimensional array with column-major ordering.
    /// \param[in]  lda
    ///             Leading dimension of two-dimensional matrix
    ///             \f$ \mathbf{A} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    ///
    /// \sa         cublasXaxpy

    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    cublasStatus_t cublasXgemv<__nv_fp8_e4m3>(
            cublasHandle_t handle,
            cublasOperation_t trans,
            int m,
            int n,
            const __nv_fp8_e4m3* RESTRICT alpha,
            const __nv_fp8_e4m3* RESTRICT A,
            int lda,
            const __nv_fp8_e4m3* RESTRICT x,
            int incx,
            const __nv_fp8_e4m3* RESTRICT beta,
            __nv_fp8_e4m3* RESTRICT y,
            int incy)
    {
        // Void unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) handle;

        cudaError_t error = cublas_impl::cublasTgemv<__nv_fp8_e4m3, float>(
                trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        
        if (error != cudaSuccess)
        {
            return CUBLAS_STATUS_SUCCESS;
        }
        else
        {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    #endif


    // ===========
    // cublasXgemv (__half)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \alpha \text{op}(\mathbf{A})
    ///             \boldsymbol{x} + \beta \boldsymbol{y} \f$.
    ///
    /// \details    This function is not a template wrapper for CuBLAS, since
    ///             CuBLAS API does not have \c cublasHgemv (where \c H is
    ///             used for \c __half type). As such, this function is
    ///             implemented with CUDA, rather than from CuBLAS.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  trans
    ///             If set to \c CUBLAS_OP_N or \c CUBLAS_OP_T, the operator
    ///             \f$ \mathbf{A} \f$ is not transposed or transposed,
    ///             respectively.
    /// \param[in]  m
    ///             Number of rows of matrix \f$ \mathbf{A} \f$.
    /// \param[in]  n
    ///             Number of columns of matrix \f$ \mathbf{A} \f$.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$.
    /// \param[in]  A
    ///             Two-dimensional matrix \f$ \mathbf{A} \f$ stored on GPU
    ///             device as one-dimensional array with column-major ordering.
    /// \param[in]  lda
    ///             Leading dimension of two-dimensional matrix
    ///             \f$ \mathbf{A} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    ///
    /// \sa         cublasXaxpy

    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    cublasStatus_t cublasXgemv<__half>(
            cublasHandle_t handle,
            cublasOperation_t trans,
            int m,
            int n,
            const __half* RESTRICT alpha,
            const __half* RESTRICT A,
            int lda,
            const __half* RESTRICT x,
            int incx,
            const __half* RESTRICT beta,
            __half* RESTRICT y,
            int incy)
    {
        // Void unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) handle;

        cudaError_t error = cublas_impl::cublasTgemv<__half, float>(
                trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        
        if (error != cudaSuccess)
        {
            return CUBLAS_STATUS_SUCCESS;
        }
        else
        {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    #endif


    // ===========
    // cublasXgemv (__nv_bfloat16)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \alpha \text{op}(\mathbf{A})
    ///             \boldsymbol{x} + \beta \boldsymbol{y} \f$.
    ///
    /// \details    This function is not a template wrapper for CuBLAS, since
    ///             CuBLAS API does not have \c cublasHgemv (where \c H is
    ///             used for \c __nv_bfloat16 type). As such, this function is
    ///             implemented with CUDA, rather than from CuBLAS.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  trans
    ///             If set to \c CUBLAS_OP_N or \c CUBLAS_OP_T, the operator
    ///             \f$ \mathbf{A} \f$ is not transposed or transposed,
    ///             respectively.
    /// \param[in]  m
    ///             Number of rows of matrix \f$ \mathbf{A} \f$.
    /// \param[in]  n
    ///             Number of columns of matrix \f$ \mathbf{A} \f$.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$.
    /// \param[in]  A
    ///             Two-dimensional matrix \f$ \mathbf{A} \f$ stored on GPU
    ///             device as one-dimensional array with column-major ordering.
    /// \param[in]  lda
    ///             Leading dimension of two-dimensional matrix
    ///             \f$ \mathbf{A} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    ///
    /// \sa         cublasXaxpy

    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    cublasStatus_t cublasXgemv<__nv_bfloat16>(
            cublasHandle_t handle,
            cublasOperation_t trans,
            int m,
            int n,
            const __nv_bfloat16* RESTRICT alpha,
            const __nv_bfloat16* RESTRICT A,
            int lda,
            const __nv_bfloat16* RESTRICT x,
            int incx,
            const __nv_bfloat16* RESTRICT beta,
            __nv_bfloat16* RESTRICT y,
            int incy)
    {
        // Void unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) handle;

        cudaError_t error = cublas_impl::cublasTgemv<__nv_bfloat16, float>(
                trans, m, n, alpha, A, lda, x, incx, beta, y, incy);

        if (error != cudaSuccess)
        {
            return CUBLAS_STATUS_SUCCESS;
        }
        else
        {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    #endif


    // ===========
    // cublasXgemv (float)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \alpha \text{op}(\mathbf{A})
    ///             \boldsymbol{x} + \beta \boldsymbol{y} \f$.
    ///
    /// \details    This function is a template wrapper for \c cublasSgemv.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  trans
    ///             If set to \c CUBLAS_OP_N or \c CUBLAS_OP_T, the operator
    ///             \f$ \mathbf{A} \f$ is not transposed or transposed,
    ///             respectively.
    /// \param[in]  m
    ///             Number of rows of matrix \f$ \mathbf{A} \f$.
    /// \param[in]  n
    ///             Number of columns of matrix \f$ \mathbf{A} \f$.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$.
    /// \param[in]  A
    ///             Two-dimensional matrix \f$ \mathbf{A} \f$ stored on GPU
    ///             device as one-dimensional array with column-major ordering.
    /// \param[in]  lda
    ///             Leading dimension of two-dimensional matrix
    ///             \f$ \mathbf{A} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    ///
    /// \sa         cublasXaxpy

    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    cublasStatus_t cublasXgemv<float>(
            cublasHandle_t handle,
            cublasOperation_t trans,
            int m,
            int n,
            const float* RESTRICT alpha,
            const float* RESTRICT A,
            int lda,
            const float* RESTRICT x,
            int incx,
            const float* RESTRICT beta,
            float* RESTRICT y,
            int incy)
    {
        
        #if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
            // Use in-house implementation
            cudaError_t error = cublas_impl::cublasTgemv<float, float>(
                    trans, m, n, alpha, A, lda, x, incx, beta, y, incy);

            if (error != cudaSuccess)
            {
                return CUBLAS_STATUS_SUCCESS;
            }
            else
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }

        #else
            // Use Nvidia's CuBLAS
            return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx,
                               beta, y, incy);
        #endif
    }
    #endif


    // ===========
    // cublasXgemv (double)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \alpha \text{op}(\mathbf{A})
    ///             \boldsymbol{x} + \beta \boldsymbol{y} \f$.
    ///
    /// \details    This function is a template wrapper for \c cublasDgemv.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  trans
    ///             If set to \c CUBLAS_OP_N or \c CUBLAS_OP_T, the operator
    ///             \f$ \mathbf{A} \f$ is not transposed or transposed,
    ///             respectively.
    /// \param[in]  m
    ///             Number of rows of matrix \f$ \mathbf{A} \f$.
    /// \param[in]  n
    ///             Number of columns of matrix \f$ \mathbf{A} \f$.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$.
    /// \param[in]  A
    ///             Two-dimensional matrix \f$ \mathbf{A} \f$ stored on GPU
    ///             device as one-dimensional array with column-major ordering.
    /// \param[in]  lda
    ///             Leading dimension of two-dimensional matrix
    ///             \f$ \mathbf{A} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    ///
    /// \sa         cublasXaxpy

    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    cublasStatus_t cublasXgemv<double>(
            cublasHandle_t handle,
            cublasOperation_t trans,
            int m,
            int n,
            const double* RESTRICT alpha,
            const double* RESTRICT A,
            int lda,
            const double* RESTRICT x,
            int incx,
            const double* RESTRICT beta,
            double* RESTRICT y,
            int incy)
    {
        #if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
            // Use in-house implementation
            cudaError_t error = cublas_impl::cublasTgemv<double, double>(
                    trans, m, n, alpha, A, lda, x, incx, beta, y, incy);

            if (error != cudaSuccess)
            {
                return CUBLAS_STATUS_SUCCESS;
            }
            else
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }

        #else
            // Use Nvidia's CuBLAS
            return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx,
                               beta, y, incy);
        #endif
    }
    #endif


    // ===========
    // cublasXcopy (__half)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \boldsymbol{x} \f$ in
    ///             \c __half type.
    ///
    /// \details    This function is not a template wrapper for CuBLAS, since
    ///             CuBLAS API does not have \c cublasHcopy (where \c H is
    ///             used for \c __half type). As such, this function is
    ///             implemented with CUDA, rather than from CuBLAS.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  n
    ///             Size of the array \f$ \boldsymbol{x} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    ///
    /// \sa         cublasXaxpy

    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    cublasStatus_t cublasXcopy<__half>(
            cublasHandle_t handle,
            int n,
            const __half* RESTRICT x,
            int incx,
            __half* RESTRICT y,
            int incy)
    {
        // Void unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) handle;

        cudaError_t error = cublas_impl::cublasTcopy<__half>(
                n, x, incx, y, incy);

        if (error != cudaSuccess)
        {
            return CUBLAS_STATUS_SUCCESS;
        }
        else
        {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    #endif


    // ===========
    // cublasXcopy (__nv_bfloat16)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \boldsymbol{x} \f$ in
    ///             \c __nv_bfloat16 type.
    ///
    /// \details    This function is not a template wrapper for CuBLAS, since
    ///             CuBLAS API does not have \c cublasHcopy (where \c H is
    ///             used for \c __nv_bfloat16 type). As such, this function is
    ///             implemented with CUDA, rather than from CuBLAS.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  n
    ///             Size of the array \f$ \boldsymbol{x} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    ///
    /// \sa         cublasXaxpy

    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    cublasStatus_t cublasXcopy<__nv_bfloat16>(
            cublasHandle_t handle,
            int n,
            const __nv_bfloat16* RESTRICT x,
            int incx,
            __nv_bfloat16* RESTRICT y,
            int incy)
    {
        // Void unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) handle;

        cudaError_t error =  cublas_impl::cublasTcopy<__nv_bfloat16>(
                n, x, incx, y, incy);

        if (error != cudaSuccess)
        {
            return CUBLAS_STATUS_SUCCESS;
        }
        else
        {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    #endif


    // ===========
    // cublasXcopy (float)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \boldsymbol{x} \f$ in
    ///             \c float type.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  n
    ///             Size of the array \f$ \boldsymbol{x} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    ///
    /// \sa         cublasXaxpy

    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    cublasStatus_t cublasXcopy<float>(
            cublasHandle_t handle,
            int n,
            const float* RESTRICT x,
            int incx,
            float* RESTRICT y,
            int incy)
    {
        #if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
            // Use in-house implementation
            cudaError_t error = cublas_impl::cublasTcopy<float>(
                    n, x, incx, y, incy);
            
            if (error != cudaSuccess)
            {
                return CUBLAS_STATUS_SUCCESS;
            }
            else
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }

        #else
            // Use Nvidia's CuBLAS
            return cublasScopy(handle, n, x, incx, y, incy);
        #endif
    }
    #endif


    // ===========
    // cublasXcopy (double)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \boldsymbol{x} \f$ in
    ///             \c double type.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  n
    ///             Size of the array \f$ \boldsymbol{x} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    ///
    /// \sa         cublasXaxpy

    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    cublasStatus_t cublasXcopy<double>(
            cublasHandle_t handle,
            int n,
            const double* RESTRICT x,
            int incx,
            double* RESTRICT y,
            int incy)
    {
        #if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
            // Use in-house implementation
            cudaError_t error = cublas_impl::cublasTcopy<double>(
                    n, x, incx, y, incy);

            if (error != cudaSuccess)
            {
                return CUBLAS_STATUS_SUCCESS;
            }
            else
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }

        #else
            // Use Nvidia's CuBLAS
            return cublasDcopy(handle, n, x, incx, y, incy);
        #endif
    }
    #endif


    // ===========
    // cublasXaxpy (__half)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \alpha \boldsymbol{x} +
    ///             \boldsymbol{y} \f$ on \c __half precision.
    ///
    /// \details    This function is a \c half type implementation similar
    ///             to cuBLAS's \c cublasSaxpy.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  n
    ///             Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    ///
    /// \sa         cublasXgemv

    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    cublasStatus_t cublasXaxpy<__half>(
            cublasHandle_t handle,
            int n,
            const __half* RESTRICT alpha,
            const __half* RESTRICT x,
            int incx,
            __half* RESTRICT y,
            int incy)
    {
        // Void unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) handle;

        cudaError_t error = cublas_impl::cublasTaxpy<__half>(
                n, alpha, x, incx, y, incy);

        if (error != cudaSuccess)
        {
            return CUBLAS_STATUS_SUCCESS;
        }
        else
        {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    #endif


    // ===========
    // cublasXaxpy (__nv_bfloat16)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \alpha \boldsymbol{x} +
    ///             \boldsymbol{y} \f$ on \c __nv_bfloat16 precision.
    ///
    /// \details    This function is a \c half type implementation similar
    ///             to cuBLAS's \c cublasSaxpy.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  n
    ///             Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    ///
    /// \sa         cublasXgemv

    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    cublasStatus_t cublasXaxpy<__nv_bfloat16>(
            cublasHandle_t handle,
            int n,
            const __nv_bfloat16* RESTRICT alpha,
            const __nv_bfloat16* RESTRICT x,
            int incx,
            __nv_bfloat16* RESTRICT y,
            int incy)
    {
        // Void unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) handle;

        cudaError_t error = cublas_impl::cublasTaxpy<__nv_bfloat16>(
                n, alpha, x, incx, y, incy);

        if (error != cudaSuccess)
        {
            return CUBLAS_STATUS_SUCCESS;
        }
        else
        {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    #endif


    // ===========
    // cublasXaxpy (float)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \alpha \boldsymbol{x} +
    ///             \boldsymbol{y} \f$ on \c float precision.
    ///
    /// \details    This function is a \c half type implementation similar
    ///             to cuBLAS's \c cublasSaxpy.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  n
    ///             Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    ///
    /// \sa         cublasXgemv

    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    cublasStatus_t cublasXaxpy<float>(
            cublasHandle_t handle,
            int n,
            const float* RESTRICT alpha,
            const float* RESTRICT x,
            int incx,
            float* RESTRICT y,
            int incy)
    {
        #if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
            // Use in-house implementation
            cudaError_t error = cublas_impl::cublasTaxpy<float>(
                    n, alpha, x, incx, y, incy);

            if (error != cudaSuccess)
            {
                return CUBLAS_STATUS_SUCCESS;
            }
            else
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }

        #else
            return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
        #endif
    }
    #endif


    // ===========
    // cublasXaxpy (double)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \alpha \boldsymbol{x} +
    ///             \boldsymbol{y} \f$ on \c double precision.
    ///
    /// \details    This function is a \c half type implementation similar
    ///             to cuBLAS's \c cublasSaxpy.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  n
    ///             Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    ///
    /// \sa         cublasXgemv

    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    cublasStatus_t cublasXaxpy<double>(
            cublasHandle_t handle,
            int n,
            const double* RESTRICT alpha,
            const double* RESTRICT x,
            int incx,
            double* RESTRICT y,
            int incy)
    {
        #if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
            // Use in-house implementation
            cudaError_t error = cublas_impl::cublasTaxpy<double>(
                    n, alpha, x, incx, y, incy);

            if (error != cudaSuccess)
            {
                return CUBLAS_STATUS_SUCCESS;
            }
            else
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }

        #else
            return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
        #endif
    }
    #endif


    // ==========
    // cublasXdot (__half)
    // ==========

    /// \brief      Performs \f$ \boldsymbol{y} = \boldsymbol{x} \cdot
    ///             \boldsymbol{y} \f$ on \c __half precision.
    ///
    /// \details    This function is a \c half type implementation similar
    ///             to cuBLAS's \c cublasSdot.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  n
    ///             Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    /// \param[out] result
    ///             The dot product of two vectors.
    ///
    /// \sa         cublasHaxpy

    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    cublasStatus_t cublasXdot<__half>(
            cublasHandle_t handle,
            int n,
            const __half* RESTRICT x,
            int incx,
            const __half* RESTRICT y,
            int incy,
            __half* RESTRICT result)
    {
        // Void unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) handle;

        cudaError_t error = cublas_impl::cublasTdot<__half, float>(
                n, x, incx, y, incy, result);

        if (error != cudaSuccess)
        {
            return CUBLAS_STATUS_SUCCESS;
        }
        else
        {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    #endif


    // ==========
    // cublasXdot (__nv_bfloat16)
    // ==========

    /// \brief      Performs \f$ \boldsymbol{y} = \boldsymbol{x} \cdot
    ///             \boldsymbol{y} \f$ on \c __nv_bfloat16 precision.
    ///
    /// \details    This function is a \c half type implementation similar
    ///             to cuBLAS's \c cublasSdot.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  n
    ///             Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    /// \param[out] result
    ///             The dot product of two vectors.
    ///
    /// \sa         cublasHaxpy

    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    cublasStatus_t cublasXdot<__nv_bfloat16>(
            cublasHandle_t handle,
            int n,
            const __nv_bfloat16* RESTRICT x,
            int incx,
            const __nv_bfloat16* RESTRICT y,
            int incy,
            __nv_bfloat16* RESTRICT result)
    {
        // Void unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) handle;

        cudaError_t error = cublas_impl::cublasTdot<__nv_bfloat16, float>(
                n, x, incx, y, incy, result);

        if (error != cudaSuccess)
        {
            return CUBLAS_STATUS_SUCCESS;
        }
        else
        {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    #endif


    // ==========
    // cublasXdot (float)
    // ==========

    /// \brief      Performs \f$ \boldsymbol{y} = \boldsymbol{x} \cdot
    ///             \boldsymbol{y} \f$ on \c float precision.
    ///
    /// \details    This function is a \c half type implementation similar
    ///             to cuBLAS's \c cublasSdot.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  n
    ///             Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    /// \param[out] result
    ///             The dot product of two vectors.
    ///
    /// \sa         cublasHaxpy

    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    cublasStatus_t cublasXdot<float>(
            cublasHandle_t handle,
            int n,
            const float* RESTRICT x,
            int incx,
            const float* RESTRICT y,
            int incy,
            float* RESTRICT result)
    {
        #if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
            // Use in-house implementation
            cudaError_t error = cublas_impl::cublasTdot<float, float>(
                    n, x, incx, y, incy, result);

            if (error != cudaSuccess)
            {
                return CUBLAS_STATUS_SUCCESS;
            }
            else
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }

        #else
            return cublasSdot(handle, n, x, incx, y, incy, result);
        #endif
    }
    #endif


    // ==========
    // cublasXdot (double)
    // ==========

    /// \brief      Performs \f$ \boldsymbol{y} = \boldsymbol{x} \cdot
    ///             \boldsymbol{y} \f$ on \c double precision.
    ///
    /// \details    This function is a \c half type implementation similar
    ///             to cuBLAS's \c cublasSdot.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  n
    ///             Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] y
    ///             Output vector \f$ \boldsymbol{y} \f$ stored on GPU device.
    /// \param[in]  incy
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{y} \f$.
    /// \param[out] result
    ///             The dot product of two vectors.
    ///
    /// \sa         cublasHaxpy

    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    cublasStatus_t cublasXdot<double>(
            cublasHandle_t handle,
            int n,
            const double* RESTRICT x,
            int incx,
            const double* RESTRICT y,
            int incy,
            double* RESTRICT result)
    {
        #if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
            // Use in-house implementation
            cudaError_t error = cublas_impl::cublasTdot<double, double>(
                    n, x, incx, y, incy, result);

            if (error != cudaSuccess)
            {
                return CUBLAS_STATUS_SUCCESS;
            }
            else
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }

        #else
            return cublasDdot(handle, n, x, incx, y, incy, result);
        #endif
    }
    #endif


    // ===========
    // cublasXnrm2 (__half)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \boldsymbol{x} \cdot
    ///             \boldsymbol{x} \f$ on \c __half precision.
    ///
    /// \details    This function is a \c half type implementation similar
    ///             to cuBLAS's \c cublasSnrm2.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  n
    ///             Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] result
    ///             The norm squared of a vector.
    ///
    /// \sa         cublasHdot

    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    cublasStatus_t cublasXnrm2<__half>(
            cublasHandle_t handle,
            int n,
            const __half* RESTRICT x,
            int incx,
            __half* RESTRICT result)
    {
        // Void unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) handle;

        cudaError_t error = cublas_impl::cublasTnrm2<__half, float>(
                n, x, incx, result);

        if (error != cudaSuccess)
        {
            return CUBLAS_STATUS_SUCCESS;
        }
        else
        {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    #endif


    // ===========
    // cublasXnrm2 (__nv_bfloat16)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \boldsymbol{x} \cdot
    ///             \boldsymbol{x} \f$ on \c __nv_bfloat16 precision.
    ///
    /// \details    This function is a \c half type implementation similar
    ///             to cuBLAS's \c cublasSnrm2.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  n
    ///             Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] result
    ///             The norm squared of a vector.
    ///
    /// \sa         cublasHdot

    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    cublasStatus_t cublasXnrm2<__nv_bfloat16>(
            cublasHandle_t handle,
            int n,
            const __nv_bfloat16* RESTRICT x,
            int incx,
            __nv_bfloat16* RESTRICT result)
    {
        // Void unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) handle;

        cudaError_t error = cublas_impl::cublasTnrm2<__nv_bfloat16, float>(
                n, x, incx, result);

        if (error != cudaSuccess)
        {
            return CUBLAS_STATUS_SUCCESS;
        }
        else
        {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    #endif


    // ===========
    // cublasXnrm2 (float)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \boldsymbol{x} \cdot
    ///             \boldsymbol{x} \f$ on \c float precision.
    ///
    /// \details    This function is a \c half type implementation similar
    ///             to cuBLAS's \c cublasSnrm2.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  n
    ///             Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] result
    ///             The norm squared of a vector.
    ///
    /// \sa         cublasHdot

    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    cublasStatus_t cublasXnrm2<float>(
            cublasHandle_t handle,
            int n,
            const float* RESTRICT x,
            int incx,
            float* RESTRICT result)
    {
        #if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
            // Use in-house implementation
            cudaError_t error = cublas_impl::cublasTnrm2<float, float>(
                    n, x, incx, result);

            if (error != cudaSuccess)
            {
                return CUBLAS_STATUS_SUCCESS;
            }
            else
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }

        #else
            return cublasSnrm2(handle, n, x, incx, result);
        #endif
    }
    #endif


    // ===========
    // cublasXnrm2 (double)
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \boldsymbol{x} \cdot
    ///             \boldsymbol{x} \f$ on \c double precision.
    ///
    /// \details    This function is a \c half type implementation similar
    ///             to cuBLAS's \c cublasSnrm2.
    ///
    /// \param[in]  handle
    ///             Handle object for CuBLAS library context.
    /// \param[in]  n
    ///             Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] result
    ///             The norm squared of a vector.
    ///
    /// \sa         cublasHdot

    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    cublasStatus_t cublasXnrm2<double>(
            cublasHandle_t handle,
            int n,
            const double* RESTRICT x,
            int incx,
            double* RESTRICT result)
    {
        #if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
            // Use in-house implementation
            cudaError_t error = cublas_impl::cublasTnrm2<double, double>(
                    n, x, incx, result);

            if (error != cudaSuccess)
            {
                return CUBLAS_STATUS_SUCCESS;
            }
            else
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }

        #else
            return cublasDnrm2(handle, n, x, incx, result);
        #endif
    }
    #endif


    // ===========
    // cublasXscal (__half)
    // ===========

    /// \brief           Performs \f$ \boldsymbol{x} = \alpha \boldsymbol{x}
    ///                  \f$ on \c __half precision.
    ///
    /// \details         This function is a \c half type implementation similar
    ///                  to cuBLAS's \c cublasSscal.
    ///
    /// \param[in]       handle
    ///                  Handle object for CuBLAS library context.
    /// \param[in]       n
    ///                  Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]       alpha
    ///                  The scalar parameter \f$ \alpha \f$.
    /// \param[in, out]  x
    ///                  Input and output vector \f$ \boldsymbol{x} \f$ stored
    ///                  on GPU device. This vector is written in-place.
    /// \param[in]       incx
    ///                  Stride between consecutive elements of
    ///                  \f$ \boldsymbol{x} \f$.
    ///
    /// \sa              cublasHcopy

    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    cublasStatus_t cublasXscal<__half>(
            cublasHandle_t handle,
            int n,
            const __half* RESTRICT alpha,
            __half* RESTRICT x,
            int incx)
    {
        // Void unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) handle;

        cudaError_t error = cublas_impl::cublasTscal<__half>(
                n, alpha, x, incx);

        if (error != cudaSuccess)
        {
            return CUBLAS_STATUS_SUCCESS;
        }
        else
        {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    #endif


    // ===========
    // cublasXscal (__nv_bfloat16)
    // ===========

    /// \brief           Performs \f$ \boldsymbol{x} = \alpha \boldsymbol{x}
    ///                  \f$ on \c __nv_bfloat16 precision.
    ///
    /// \details         This function is a \c half type implementation similar
    ///                  to cuBLAS's \c cublasSscal.
    ///
    /// \param[in]       handle
    ///                  Handle object for CuBLAS library context.
    /// \param[in]       n
    ///                  Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]       alpha
    ///                  The scalar parameter \f$ \alpha \f$.
    /// \param[in, out]  x
    ///                  Input and output vector \f$ \boldsymbol{x} \f$ stored
    ///                  on GPU device. This vector is written in-place.
    /// \param[in]       incx
    ///                  Stride between consecutive elements of
    ///                  \f$ \boldsymbol{x} \f$.
    ///
    /// \sa              cublasHcopy

    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    cublasStatus_t cublasXscal<__nv_bfloat16>(
            cublasHandle_t handle,
            int n,
            const __nv_bfloat16* RESTRICT alpha,
            __nv_bfloat16* RESTRICT x,
            int incx)
    {
        // Void unused variables to avoid compiler warnings
        // (-Wno-unused-parameter)
        (void) handle;

        cudaError_t error = cublas_impl::cublasTscal<__nv_bfloat16>(
                n, alpha, x, incx);

        if (error != cudaSuccess)
        {
            return CUBLAS_STATUS_SUCCESS;
        }
        else
        {
            return CUBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    #endif


    // ===========
    // cublasXscal (float)
    // ===========

    /// \brief           Performs \f$ \boldsymbol{x} = \alpha \boldsymbol{x}
    ///                  \f$ on \c float precision.
    ///
    /// \details         This function is a \c half type implementation similar
    ///                  to cuBLAS's \c cublasSscal.
    ///
    /// \param[in]       handle
    ///                  Handle object for CuBLAS library context.
    /// \param[in]       n
    ///                  Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]       alpha
    ///                  The scalar parameter \f$ \alpha \f$.
    /// \param[in, out]  x
    ///                  Input and output vector \f$ \boldsymbol{x} \f$ stored
    ///                  on GPU device. This vector is written in-place.
    /// \param[in]       incx
    ///                  Stride between consecutive elements of
    ///                  \f$ \boldsymbol{x} \f$.
    ///
    /// \sa              cublasHcopy

    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    cublasStatus_t cublasXscal<float>(
            cublasHandle_t handle,
            int n,
            const float* RESTRICT alpha,
            float* RESTRICT x,
            int incx)
    {
        #if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
            // Use in-house implementation
            cudaError_t error = cublas_impl::cublasTscal<float>(
                    n, alpha, x, incx);

            if (error != cudaSuccess)
            {
                return CUBLAS_STATUS_SUCCESS;
            }
            else
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }

        #else
            return cublasSscal(handle, n, alpha, x, incx);
        #endif
    }
    #endif


    // ===========
    // cublasXscal (double)
    // ===========

    /// \brief           Performs \f$ \boldsymbol{x} = \alpha \boldsymbol{x}
    ///                  \f$ on \c double precision.
    ///
    /// \details         This function is a \c half type implementation similar
    ///                  to cuBLAS's \c cublasSscal.
    ///
    /// \param[in]       handle
    ///                  Handle object for CuBLAS library context.
    /// \param[in]       n
    ///                  Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]       alpha
    ///                  The scalar parameter \f$ \alpha \f$.
    /// \param[in, out]  x
    ///                  Input and output vector \f$ \boldsymbol{x} \f$ stored
    ///                  on GPU device. This vector is written in-place.
    /// \param[in]       incx
    ///                  Stride between consecutive elements of
    ///                  \f$ \boldsymbol{x} \f$.
    ///
    /// \sa              cublasHcopy

    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    cublasStatus_t cublasXscal<double>(
            cublasHandle_t handle,
            int n,
            const double* RESTRICT alpha,
            double* RESTRICT x,
            int incx)
    {
        #if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
            // Use in-house implementation
            cudaError_t error = cublas_impl::cublasTscal<double>(
                    n, alpha, x, incx);

            if (error != cudaSuccess)
            {
                return CUBLAS_STATUS_SUCCESS;
            }
            else
            {
                return CUBLAS_STATUS_INTERNAL_ERROR;
            }

        #else
            return cublasDscal(handle, n, alpha, x, incx);
        #endif
    }
    #endif

}  // namespace cublas_api
