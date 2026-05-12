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

#include "./cublas_impl.h"
#include "./cublas_impl_kernels.h"  // cublas_impl_kernels
#include "../_cu_arithmetics/cu_arithmetics.h"  // cu_arithmetics
#include <cuda_runtime.h>
#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __nv_bfloat16
#include <stdexcept>  // std::invalid_argument


// ===========
// cublas impl
// ===========

namespace cublas_impl
{
    // ===========
    // cublasTgemv
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \alpha \text{op}(\mathbf{A})
    ///             \boldsymbol{x} + \beta \boldsymbol{y} \f$.
    ///
    /// \details    This function is a custom implementation of cuBLAS's
    ///             \c cublasSgemv from scratch. The corresponding kernel
    ///             code can be found at
    ///             \ref cublas_impl_kernels::cublasTgemv_kernel .
    ///
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
    /// \return     error
    ///             CUDA synchronize error code.
    ///
    /// \sa         cublas_impl_kernels::cublasTgemv_kernel

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
            int incy)
    {
        // Determine array sizes based on operation of A
        bool trans_;
        int x_size;
        int y_size;

        if (trans == CUBLAS_OP_N)
        {
            // A is not transposed
            trans_ = false;
            y_size = m;
            x_size = n;
        }
        else if (trans == CUBLAS_OP_T)
        {
            // A is transposed
            trans_ = true;
            y_size = n;
            x_size = m;
        }
        else
        {
            throw std::invalid_argument(
                "'trans' argument must be CUBLAS_OP_N or CUBLAS_OP_T.");
        }

        // The optimal number of threads per block (here 640) is obtained by
        // calling cudaOccupancyMaxPotentialBlockSize() in a separate
        // benchmark.
        const int threads_per_block = 640;
        dim3 dim_block(threads_per_block);

        // We assume each thread represents one element of y. That is, the
        // total number of threads is the size of y.
        int blocks_per_grid = \
                (y_size + threads_per_block - 1) / threads_per_block;
        dim3 dim_grid(blocks_per_grid);

        // Calling kernel code
        cublas_impl_kernels::cublasTgemv_kernel<
            DataType, ComputeType, threads_per_block>
            <<<dim_grid, dim_block>>>(
                    trans_, y_size, x_size, *alpha, A, lda, x, incx, *beta, y,
                    incy);

        cudaError_t error = cudaDeviceSynchronize();
     
        return error;
    }


    // ===========
    // cublasTcopy
    // ===========

    /// \brief      Performs \f$ \boldsymbol{y} = \boldsymbol{x} \f$.
    ///
    /// \details    This function is a custom implementation of cuBLAS's
    ///             \c cublasScopy from scratch. The corresponding kernel
    ///             code can be found at
    ///             \ref cublas_impl_kernels::cublasTcopy_kernel .
    ///
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
    /// \return     error
    ///             CUDA synchronize error code.
    ///
    /// \sa         cublas_impl_kernels::cublasTcopy_kernel
    
    template <typename DataType>
    cudaError_t cublasTcopy(
            int n,
            const DataType* RESTRICT x,
            int incx,
            DataType* RESTRICT y,
            int incy)
    {
        // Set number of device threads and blocks
        const int threads_per_block = 256;
        int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

        // Call device code
        cublas_impl_kernels::cublasTcopy_kernel<DataType><<<
            blocks_per_grid, threads_per_block>>>(
                n, x, incx, y, incy);

        cudaError_t error = cudaDeviceSynchronize();

        return error;
    }


    // ===========
    // cublasTaxpy
    // ===========
    
    /// \brief      Performs \f$ \boldsymbol{y} = \alpha \boldsymbol{x} +
    ///             \boldsymbol{y} \f$.
    ///
    /// \details    This function is a custom implementation of cuBLAS's
    ///             \c cublasSaxpy from scratch. The corresponding kernel
    ///             code can be found at
    ///             \ref cublas_impl_kernels::cublasTaxpy_kernel .
    ///
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
    /// \return     error
    ///             CUDA synchronize error code.
    ///
    /// \sa         cublas_impl_kernels::cublasTaxpy_kernel
    
    template <typename DataType>
    cudaError_t cublasTaxpy(
            int n,
            const DataType* RESTRICT alpha,
            const DataType* RESTRICT x,
            int incx,
            DataType* RESTRICT y,
            int incy)
    {
        // Set number of device threads and blocks
        const int threads_per_block = 256;
        int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

        // Call device code
        cublas_impl_kernels::cublasTaxpy_kernel<DataType><<<
            blocks_per_grid, threads_per_block>>>(
                n, *alpha, x, incx, y, incy);

        cudaError_t error = cudaDeviceSynchronize();

        return error;
    }


    // ==========
    // cublasTdot
    // ==========
    
    /// \brief      Computes \f$ a = \boldsymbol{x} \cdot \boldsymbol{y} \f$.
    ///
    /// \details    This function is a custom implementation of cuBLAS's
    ///             \c cublasSdot from scratch. The corresponding kernel
    ///             code can be found at
    ///             \ref cublas_impl_kernels::cublasTdot_kernel .
    ///
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
    /// \return     error
    ///             CUDA synchronize error code.
    ///
    /// \sa         cublas_impl_kernels::cublasTdot_kernel
    
    template <typename DataType, typename ComputeType>
    cudaError_t cublasTdot(
            int n,
            const DataType* RESTRICT x,
            int incx,
            const DataType* RESTRICT y,
            int incy,
            DataType* RESTRICT result)
    {
        // device pointer to store the result (this is a scalar value)
        ComputeType *device_result;
        cudaMalloc(&device_result, sizeof(ComputeType));
        cudaMemset(device_result, static_cast<ComputeType>(0.0f),
                   sizeof(ComputeType));

        // Set number of device threads and blocks
        const int threads_per_block = 256;
        int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

        // Call device code
        cublas_impl_kernels::cublasTdot_kernel<
            DataType, ComputeType, threads_per_block><<<
            blocks_per_grid, threads_per_block>>>(
                n, x, incx, y, incy, device_result);

        cudaError_t error = cudaDeviceSynchronize();

        // Return back result from device and store as higher precision type
        ComputeType host_result_comp;
        cudaMemcpy(&host_result_comp, device_result, sizeof(ComputeType),
                   cudaMemcpyDeviceToHost);

        // Convert type to match output type
        *result = cu_arithmetics::cast<ComputeType, DataType>(
                host_result_comp);

        cudaFree(device_result);

        return error;
    }


    // ===========
    // cublasTnrm2
    // ===========
    
    /// \brief      Computes \f$ a = \boldsymbol{x} \cdot \boldsymbol{x} \f$.
    ///
    /// \details    This function is a custom implementation of cuBLAS's
    ///             \c cublasSnrm2 from scratch. The corresponding kernel
    ///             code can be found at
    ///             \ref cublas_impl_kernels::cublasTnrm2_kernel .
    ///
    /// \param[in]  n
    ///             Size of array \f$ \boldsymbol{x} \f$.
    /// \param[in]  x
    ///             Input vector \f$ \boldsymbol{x} \f$ stored on GPU device.
    /// \param[in]  incx
    ///             Stride between consecutive elements of
    ///             \f$ \boldsymbol{x} \f$.
    /// \param[out] result
    ///             The norm squared of a vector.
    /// \return     error
    ///             CUDA synchronize error code.
    ///
    /// \sa         cublas_impl_kernels::cublasTnrm2_kernel
    
    template <typename DataType, typename ComputeType>
    cudaError_t cublasTnrm2(
            int n,
            const DataType* RESTRICT x,
            int incx,
            DataType* RESTRICT result)
    {
        // device pointer to store the result (this is a scalar value)
        ComputeType *device_result;
        cudaMalloc(&device_result, sizeof(ComputeType));
        cudaMemset(device_result, static_cast<ComputeType>(0.0f),
                   sizeof(ComputeType));

        // Set number of device threads and blocks
        const int threads_per_block = 256;
        int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

        // Call device code
        cublas_impl_kernels::cublasTnrm2_kernel<
            DataType, ComputeType, threads_per_block><<<
            blocks_per_grid, threads_per_block>>>(
                n, x, incx, device_result);

        cudaError_t error = cudaDeviceSynchronize();

        // Return back result from device and store as higher precision type
        ComputeType host_result_comp;
        cudaMemcpy(&host_result_comp, device_result, sizeof(ComputeType),
                   cudaMemcpyDeviceToHost);

        // Convert type to match output type
        *result = cu_arithmetics::cast<ComputeType, DataType>(
                host_result_comp);

        cudaFree(device_result);

        return error;
    }


    // ===========
    // cublasTscal
    // ===========
    
    /// \brief           Performs \f$ \boldsymbol{x} = \alpha \boldsymbol{x}
    ///                  \f$.
    ///
    /// \details         This function is a custom implementation of cuBLAS's
    ///                  \c cublasSscale from scratch. The corresponding kernel
    ///                  code can be found at
    ///                  \ref cublas_impl_kernels::cublasTscal_kernel .
    ///
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
    /// \return          error
    ///                  CUDA synchronize error code.
    ///
    /// \sa              cublas_impl_kernels::cublasTscal_kernel
    
    template <typename DataType>
    cudaError_t cublasTscal(
            int n,
            const DataType* RESTRICT alpha,
            DataType* RESTRICT x,
            int incx)
    {
        // Set number of device threads and blocks
        int threads_per_block = 256;
        int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

        // Call device code
        cublas_impl_kernels::cublasTscal_kernel<DataType><<<
            blocks_per_grid, threads_per_block>>>(
                n, *alpha, x, incx);

        cudaError_t error = cudaDeviceSynchronize();

        return error;
    }

}  // namespace cublas_impl


// ===============================
// Explicit template instantiation
// ===============================

// cublasTgemv (__nv_fp8_e5m2)
#if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template
    cudaError_t cublas_impl::cublasTgemv<__nv_fp8_e5m2, float>(
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
            int incy);
#endif

// cublasTgemv (__nv_fp8_e4m3)
#if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template
    cudaError_t cublas_impl::cublasTgemv<__nv_fp8_e4m3, float>(
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
            int incy);
#endif

// cublasTgemv (__half)
#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template
    cudaError_t cublas_impl::cublasTgemv<__half, float>(
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
            int incy);
#endif

// cublasTgemv (__nv_bfloat16)
#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template
    cudaError_t cublas_impl::cublasTgemv<__nv_bfloat16, float>(
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
            int incy);
#endif

// cublasTgemv (float)
#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    cudaError_t cublas_impl::cublasTgemv<float, float>(
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
            int incy);
#endif
#endif

// cublasTgemv (double)
#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    cudaError_t cublas_impl::cublasTgemv<double, double>(
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
            int incy);
#endif
#endif

// cublasTcopy (__half)
#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template
    cudaError_t cublas_impl::cublasTcopy<__half>(
            int n,
            const __half* RESTRICT x,
            int incx,
            __half* RESTRICT y,
            int incy);
#endif

// cublasTcopy (__nv_bfloat16)
#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template
    cudaError_t cublas_impl::cublasTcopy<__nv_bfloat16>(
            int n,
            const __nv_bfloat16* RESTRICT x,
            int incx,
            __nv_bfloat16* RESTRICT y,
            int incy);
#endif

// cublasTcopy (float)
#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    cudaError_t cublas_impl::cublasTcopy<float>(
            int n,
            const float* RESTRICT x,
            int incx,
            float* RESTRICT y,
            int incy);
#endif
#endif

// cublasTcopy (double)
#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    cudaError_t cublas_impl::cublasTcopy<double>(
            int n,
            const double* RESTRICT x,
            int incx,
            double* RESTRICT y,
            int incy);
#endif
#endif

// cublasTaxpy (__half)
#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template
    cudaError_t cublas_impl::cublasTaxpy<__half>(
            int n,
            const __half* RESTRICT alpha,
            const __half* RESTRICT x,
            int incx,
            __half* RESTRICT y,
            int incy);
#endif

// cublasTaxpy (__nv_bfloat16)
#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template
    cudaError_t cublas_impl::cublasTaxpy<__nv_bfloat16>(
            int n,
            const __nv_bfloat16* RESTRICT alpha,
            const __nv_bfloat16* RESTRICT x,
            int incx,
            __nv_bfloat16* RESTRICT y,
            int incy);
#endif

// cublasTaxpy (float)
#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    cudaError_t cublas_impl::cublasTaxpy<float>(
            int n,
            const float* RESTRICT alpha,
            const float* RESTRICT x,
            int incx,
            float* RESTRICT y,
            int incy);
#endif
#endif

// cublasTaxpy (double)
#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    cudaError_t cublas_impl::cublasTaxpy<double>(
            int n,
            const double* RESTRICT alpha,
            const double* RESTRICT x,
            int incx,
            double* RESTRICT y,
            int incy);
#endif
#endif

// cublasTdot (__half)
#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template
    cudaError_t cublas_impl::cublasTdot<__half, float>(
            int n,
            const __half* RESTRICT x,
            int incx,
            const __half* RESTRICT y,
            int incy,
            __half* RESTRICT result);
#endif

// cublasTdot (__nv_bfloat16)
#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template
    cudaError_t cublas_impl::cublasTdot<__nv_bfloat16, float>(
            int n,
            const __nv_bfloat16* RESTRICT x,
            int incx,
            const __nv_bfloat16* RESTRICT y,
            int incy,
            __nv_bfloat16* RESTRICT result);
#endif

// cublasTdot (float)
#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    cudaError_t cublas_impl::cublasTdot<float, float>(
            int n,
            const float* RESTRICT x,
            int incx,
            const float* RESTRICT y,
            int incy,
            float* RESTRICT result);
#endif
#endif

// cublasTdot (double)
#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    cudaError_t cublas_impl::cublasTdot<double, double>(
            int n,
            const double* RESTRICT x,
            int incx,
            const double* RESTRICT y,
            int incy,
            double* RESTRICT result);
#endif
#endif

// cublasTnrm2 (__half)
#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template
    cudaError_t cublas_impl::cublasTnrm2<__half, float>(
            int n,
            const __half* RESTRICT x,
            int incx,
            __half* RESTRICT result);
#endif

// cublasTnrm2 (__nv_bfloat16)
#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template
    cudaError_t cublas_impl::cublasTnrm2<__nv_bfloat16, float>(
            int n,
            const __nv_bfloat16* RESTRICT x,
            int incx,
            __nv_bfloat16* RESTRICT result);
#endif

// cublasTnrm2 (float)
#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
template
    cudaError_t cublas_impl::cublasTnrm2<float, float>(
            int n,
            const float* RESTRICT x,
            int incx,
            float* RESTRICT result);
#endif
#endif

// cublasTnrm2 (double)
#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    cudaError_t cublas_impl::cublasTnrm2<double, double>(
            int n,
            const double* RESTRICT x,
            int incx,
            double* RESTRICT result);
#endif
#endif

// cublasTscal (__half)
#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template
    cudaError_t cublas_impl::cublasTscal<__half>(
            int n,
            const __half* RESTRICT alpha,
            __half* RESTRICT x,
            int incx);
#endif

// cublasTscal (__nv_bfloat16)
#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template
    cudaError_t cublas_impl::cublasTscal<__nv_bfloat16>(
            int n,
            const __nv_bfloat16* RESTRICT alpha,
            __nv_bfloat16* RESTRICT x,
            int incx);
#endif

// cublasTscal (float)
#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    cudaError_t cublas_impl::cublasTscal<float>(
            int n,
            const float* RESTRICT alpha,
            float* RESTRICT x,
            int incx);
#endif
#endif

// cublasTscal (double)
#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    cudaError_t cublas_impl::cublasTscal<double>(
            int n,
            const double* RESTRICT alpha,
            double* RESTRICT x,
            int incx);
#endif
#endif
