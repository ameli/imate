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

#include "./cublas_impl_kernels.h"
#include <cuda_runtime.h>
#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __nv_bfloat16
#include "./atomic_add.h"  // atomicAdd (for double precision)
#include "../_cu_arithmetics/cu_arithmetics.h"  // cu_arithmetics


// ===================
// cublas impl kernels
// ===================

namespace cublas_impl_kernels
{
    // ==================
    // cublasTgemv kernel
    // ==================

    /// \brief      Performs the operation \f$ \boldsymbol{y} = \alpha
    ///             \mathbf{A} \boldsymbol{x} + \beta \boldsymbol{y} \f$.
    ///
    /// \details    This function is the device (kernel) code for
    ///             \ref cublas_impl::cublasTgemv() .
    ///
    /// \note       This function incorporate both non-transposed and
    ///             transposed operations. To this end, here \c m and \c n are
    ///             defined based on the sizes of \c y and \c x (respectively),
    ///             not the size of \c A or its transpose. The matrix \c A
    ///             (regardless of being transposed) is \c m*n.
    ///
    /// \param[in]  trans
    ///             If set to \c CUBLAS_OP_N or \c CUBLAS_OP_T, the operator
    ///             \f$ \mathbf{A} \f$ is not transposed or transposed,
    ///             respectively.
    /// \param[in]  m
    ///             Size of \c y.
    /// \param[in]  n
    ///             Size of \c x.
    /// \param[in]  alpha
    ///             Scalar parameter \f$ \alpha \f$.
    /// \param[in]  A
    ///             Matrix \c A. The matrix is assumed to be stored as a
    ///             coalesced 1D array with column-major ordering. The matrix
    ///             size is \c m*n.
    /// \param[in]  lda
    ///             Leading dimension of \c A.
    /// \param[in]  x
    ///             Input vector \c x of size \c n*incx.
    /// \param[in]   incx
    ///              Stride between consecutive elements of
    ///              \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             Scalar parameter \f$ \beta \f$.
    /// \param[out]  y
    ///              Output vector \c y of size \c m*incy.
    /// \param[in]   incy
    ///              Stride between consecutive elements of
    ///              \f$ \boldsymbol{y} \f$.
    ///
    /// \sa          cublas_impl::cublasTgemv

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
            const int incy)
    {
        // Each thread is dedicated to compute an element of y
        const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

        // Device shared memory to cache x only (note: we do not cache A since
        // the elements of A are read only once. In contrast, x is read several
        // times).
        __shared__ DataType x_shared[block_size];

        // Summation for the dot product of i-th row of A (or A transposed)
        // with the entire x. The sum variable is local to i-th thread only,
        // and is not shared with other threads of block.
        ComputeType sum = 0.0f;

        // Iterate over blocks of x elements
        const unsigned int num_blocks = (n + block_size - 1) / block_size;

        // Each thread (index i) loops over all elements j of x in block by
        // block manner.
        #pragma unroll
        for (unsigned long int block_counter = 0;
             block_counter < num_blocks;
             ++block_counter)
        {
            // Get j-th index of x. This is only used to read x to copy it to
            // the cache of x.
            unsigned long int j = threadIdx.x + \
                block_counter * static_cast<unsigned long int>(block_size);

            // Fill x cache
            if (j < n)
            {
                // Read x from global memory to shared memory
                x_shared[threadIdx.x] = x[j * incx];
            }
            else
            {
                // If block element exceeds x size, fill cache with zeros.
                x_shared[threadIdx.x] = \
                    cu_arithmetics::cast<ComputeType, DataType>(0.0f);
            }

            // Sync all threads of block to finish caching x from global memory
            // to shared memory
            __syncthreads();

            // Now that one block of cache is filled, perform matrix-vector
            // multiplication for that one block.
            #pragma unroll
            for (unsigned int e = 0; e < block_size; ++e)
            {
                // Get the index of x (called e_j) corresponding to the e-th
                // element of the cached block. This is different than the j
                // above.
                unsigned long int e_j = e + \
                    block_counter * static_cast<unsigned long int>(block_size);

                // It is necessary to check indices i and e_j with array sizes
                // as these indices can exceed the array indices since thread
                // blocks are in the sizes of multiples of 32 (as wrap size).
                if ((i < m) && (e_j < n))
                {
                    // Perform matrix-vector multiplication for the i-th row of
                    // A (or i-th row of transposed A) and the e_j th element
                    // of x.
                    if (trans)
                    {
                        sum += cu_arithmetics::cast<DataType, ComputeType>(
                                    A[i * lda + e_j]) * \
                               cu_arithmetics::cast<DataType, ComputeType>(
                                    x_shared[e]);
                    }
                    else
                    {
                        sum += cu_arithmetics::cast<DataType, ComputeType>(
                                    A[i + e_j * lda]) * \
                               cu_arithmetics::cast<DataType, ComputeType>(
                                    x_shared[e]);
                    }
                }
            }

            // Wait till all threads of block done with their matrix-vector
            // multiplication (each thread has its own sum variable), but they
            // all read cached x. This sync barrier makes sure no thread
            // proceeds the next iteration of filling new cache.
            __syncthreads();
        }

        // Update output vector only if thread does not exceed matrix size
        if (i < m)
        {
            y[i * incy] = \
                cu_arithmetics::add<DataType>(
                    cu_arithmetics::mul<DataType>(
                        alpha,
                        cu_arithmetics::cast<ComputeType, DataType>(sum)
                    ),
                    cu_arithmetics::mul<DataType>(
                        beta,
                        y[i * incy]
                    )
                );
        }
    }


    // ==================
    // cublasTcopy kernel
    // ==================
    
    /// \brief      Performs \f$ \boldsymbol{y} = \boldsymbol{x} \f$.
    ///
    /// \details    This function is a device-code (kernel) for the host code
    ///             function for \ref cublas_impl::cublasTcopy().
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
    ///
    /// \sa         cublasTcopy
    
    template <typename DataType>
    __global__ void cublasTcopy_kernel(
            const int n,
            const DataType* RESTRICT x,
            const int incx,
            DataType* RESTRICT y,
            const int incy)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;

        if (i < n)
        {
            y[i * incy] = x[i * incx];
        }
    }


    // ==================
    // cublasTaxpy kernel
    // ==================
    
    /// \brief      Performs \f$ \boldsymbol{y} = \alpha \boldsymbol{x} +
    ///             \boldsymbol{y} \f$.
    ///
    /// \details    This function is a device-code (kernel) for the host code
    ///             function for \ref cublas_impl::cublasTaxpy().
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
    ///
    /// \sa         cublasTaxpy
    
    template <typename DataType>
    __global__ void cublasTaxpy_kernel(
            const int n,
            const DataType alpha,
            const DataType* RESTRICT x,
            const int incx,
            DataType* RESTRICT y,
            const int incy)
    {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;

        if (i < n)
        {
            y[i * incy] = \
                cu_arithmetics::add<DataType>(
                    cu_arithmetics::mul<DataType>(alpha, x[i * incx]),
                    y[i * incy]
                );
        }
    }


    // =================
    // cublasTdot kernel
    // =================
    
    /// \brief      Computes \f$ a = \boldsymbol{x} \cdot \boldsymbol{y} \f$.
    ///
    /// \details    This function is a device-code (kernel) for the host code
    ///             function for \ref cublas_impl::cublasTdot().
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
    ///
    /// \sa         cublasTdot
    
    template <
        typename DataType, typename ComputeType, unsigned int block_size>
    __global__ void cublasTdot_kernel(
            const int n,
            const DataType* RESTRICT x,
            const int incx,
            const DataType* RESTRICT y,
            const int incy,
            ComputeType* RESTRICT result)
    {
        // The size of this array should be exactly the number of blocks (for
        // this, see the corresponding host code, cublas_impl::cublasTdot)
        __shared__ ComputeType partial_sum[block_size];

        const int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        ComputeType sum = static_cast<ComputeType>(0.0f);
        while (i < n)
        {
            sum += cu_arithmetics::cast<DataType, ComputeType>(x[i * incx]) * \
                   cu_arithmetics::cast<DataType, ComputeType>(y[i * incy]);

            i += blockDim.x * gridDim.x;
        }

        partial_sum[tid] = sum;

        __syncthreads();

        // Reduction in shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                partial_sum[tid] += partial_sum[tid + stride];
            }
            __syncthreads();
        }

        // Write result for this block to global memory
        if (tid == 0)
        {
            atomicAdd(result, partial_sum[0]);
        }
    }


    // ==================
    // cublasTnrm2 kernel
    // ==================
    
    /// \brief      Computes \f$ a = \boldsymbol{x} \cdot \boldsymbol{x} \f$.
    ///
    /// \details    This function is a device-code (kernel) for the host code
    ///             function for \ref cublas_impl::cublasTnrm2().
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
    ///
    /// \sa         cublasTnrm2
    
    template <
        typename DataType, typename ComputeType, unsigned int block_size>
    __global__ void cublasTnrm2_kernel(
            const int n,
            const DataType* RESTRICT x,
            const int incx,
            ComputeType* RESTRICT result)
    {
        // The size of this array should be exactly the number of blocks (for
        // this, see the corresponding host code, cublas_impl::cublasTnrm2)
        __shared__ ComputeType partial_sum[block_size];

        const int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        ComputeType sum = static_cast<ComputeType>(0.0f);
        while (i < n)
        {
            ComputeType val = cu_arithmetics::cast<DataType, ComputeType>(
                    x[i * incx]);
            sum += val * val;
            i += blockDim.x * gridDim.x;
        }

        partial_sum[tid] = sum;

        __syncthreads();

        // Reduction in shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                partial_sum[tid] += partial_sum[tid + stride];
            }
            __syncthreads();
        }

        // Write result for this block to global memory
        if (tid == 0)
        {
            atomicAdd(result, partial_sum[0]);
        }
    }


    // ==================
    // cublasTscal kernel
    // ==================
    
    /// \brief           Performs \f$ \boldsymbol{x} = \alpha \boldsymbol{x}
    ///                  \f$.
    ///
    /// \details         This function is a device-code (kernel) for the host
    ///                  code function for \ref cublas_impl::cublasTscal().
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
    ///
    /// \sa              cublasTscal
    
    template <typename DataType>
    __global__ void cublasTscal_kernel(
            const int n,
            const DataType alpha,
            DataType* RESTRICT x,
            const int incx)
    {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;

        if (i < n)
        {
            x[i * incx] = cu_arithmetics::mul<DataType>(x[i * incx], alpha);
        }
    }

}  // namespace cublas_impl_kernels


// ===============================
// Explicit template instantiation
// ===============================

// cublasTgemv kernel (__nv_fp8_e5m2)
#if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template
    __global__ void cublas_impl_kernels::cublasTgemv_kernel<
        __nv_fp8_e5m2, float, 640>(
            const bool trans,
            const int m,
            const int n,
            const __nv_fp8_e5m2 alpha,
            const __nv_fp8_e5m2* RESTRICT A,
            const int lda,
            const __nv_fp8_e5m2* RESTRICT x,
            const int incx,
            const __nv_fp8_e5m2 beta,
            __nv_fp8_e5m2* RESTRICT y,
            const int incy);
#endif

// cublasTgemv kernel (__nv_fp8_e4m3)
#if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template
    __global__ void cublas_impl_kernels::cublasTgemv_kernel<
        __nv_fp8_e4m3, float, 640>(
            const bool trans,
            const int m,
            const int n,
            const __nv_fp8_e4m3 alpha,
            const __nv_fp8_e4m3* RESTRICT A,
            const int lda,
            const __nv_fp8_e4m3* RESTRICT x,
            const int incx,
            const __nv_fp8_e4m3 beta,
            __nv_fp8_e4m3* RESTRICT y,
            const int incy);
#endif

// cublasTgemv kernel (__half)
#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template
    __global__ void cublas_impl_kernels::cublasTgemv_kernel<
        __half, float, 640>(
            const bool trans,
            const int m,
            const int n,
            const __half alpha,
            const __half* RESTRICT A,
            const int lda,
            const __half* RESTRICT x,
            const int incx,
            const __half beta,
            __half* RESTRICT y,
            const int incy);
#endif

// cublasTgemv kernel (__nv_bfloat16)
#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template
    __global__ void cublas_impl_kernels::cublasTgemv_kernel<
        __nv_bfloat16, float, 640>(
            const bool trans,
            const int m,
            const int n,
            const __nv_bfloat16 alpha,
            const __nv_bfloat16* RESTRICT A,
            const int lda,
            const __nv_bfloat16* RESTRICT x,
            const int incx,
            const __nv_bfloat16 beta,
            __nv_bfloat16* RESTRICT y,
            const int incy);
#endif

// cublasTgemv kernel (float)
#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    __global__ void cublas_impl_kernels::cublasTgemv_kernel<
        float, float, 640>(
            const bool trans,
            const int m,
            const int n,
            const float alpha,
            const float* RESTRICT A,
            const int lda,
            const float* RESTRICT x,
            const int incx,
            const float beta,
            float* RESTRICT y,
            const int incy);
#endif
#endif

// cublasTgemv kernel (double)
#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    __global__ void cublas_impl_kernels::cublasTgemv_kernel<
        double, double, 640>(
            const bool trans,
            const int m,
            const int n,
            const double alpha,
            const double* RESTRICT A,
            const int lda,
            const double* RESTRICT x,
            const int incx,
            const double beta,
            double* RESTRICT y,
            const int incy);
#endif
#endif

// cublasTcopy kernel (__half)
#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template
    __global__ void cublas_impl_kernels::cublasTcopy_kernel<__half>(
            const int n,
            const __half* RESTRICT x,
            const int incx,
            __half* RESTRICT y,
            const int incy);
#endif

// cublasTcopy kernel (__nv_bfloat16)
#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template
    __global__ void cublas_impl_kernels::cublasTcopy_kernel<__nv_bfloat16>(
            const int n,
            const __nv_bfloat16* RESTRICT x,
            const int incx,
            __nv_bfloat16* RESTRICT y,
            const int incy);
#endif

// cublasTcopy kernel (float)
#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    __global__ void cublas_impl_kernels::cublasTcopy_kernel<float>(
            const int n,
            const float* RESTRICT x,
            const int incx,
            float* RESTRICT y,
            const int incy);
#endif
#endif

// cublasTcopy kernel (double)
#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    __global__ void cublas_impl_kernels::cublasTcopy_kernel<double>(
            const int n,
            const double* RESTRICT x,
            const int incx,
            double* RESTRICT y,
            const int incy);
#endif
#endif

// cublasTaxpy kernel (__half)
#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template
    __global__ void cublas_impl_kernels::cublasTaxpy_kernel<__half>(
            const int n,
            const __half alpha,
            const __half* RESTRICT x,
            const int incx,
            __half* RESTRICT y,
            const int incy);
#endif

// cublasTaxpy kernel (__nv_bfloat16)
#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template
    __global__ void cublas_impl_kernels::cublasTaxpy_kernel<__nv_bfloat16>(
            const int n,
            const __nv_bfloat16 alpha,
            const __nv_bfloat16* RESTRICT x,
            const int incx,
            __nv_bfloat16* RESTRICT y,
            const int incy);
#endif

// cublasTaxpy kernel (float)
#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    __global__ void cublas_impl_kernels::cublasTaxpy_kernel<float>(
            const int n,
            const float alpha,
            const float* RESTRICT x,
            const int incx,
            float* RESTRICT y,
            const int incy);
#endif
#endif

// cublasTaxpy kernel (double)
#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    __global__ void cublas_impl_kernels::cublasTaxpy_kernel<double>(
            const int n,
            const double alpha,
            const double* RESTRICT x,
            const int incx,
            double* RESTRICT y,
            const int incy);
#endif
#endif

// cublasTdot kernel (__half)
#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template
    __global__ void cublas_impl_kernels::cublasTdot_kernel<
        __half, float, 256>(
            const int n,
            const __half* RESTRICT x,
            const int incx,
            const __half* RESTRICT y,
            const int incy,
            float* RESTRICT result);
#endif

// cublasTdot kernel (__nv_bfloat16)
#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template
    __global__ void cublas_impl_kernels::cublasTdot_kernel<
        __nv_bfloat16, float, 256>(
            const int n,
            const __nv_bfloat16* RESTRICT x,
            const int incx,
            const __nv_bfloat16* RESTRICT y,
            const int incy,
            float* RESTRICT result);
#endif

// cublasTdot kernel (float)
#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    __global__ void cublas_impl_kernels::cublasTdot_kernel<
        float, float, 256>(
            const int n,
            const float* RESTRICT x,
            const int incx,
            const float* RESTRICT y,
            const int incy,
            float* RESTRICT result);
#endif
#endif

// cublasTdot kernel (double)
#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    __global__ void cublas_impl_kernels::cublasTdot_kernel<
        double, double, 256>(
            const int n,
            const double* RESTRICT x,
            const int incx,
            const double* RESTRICT y,
            const int incy,
            double* RESTRICT result);
#endif
#endif

// cublasTnrm2 kernel (__half)
#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template
    __global__ void cublas_impl_kernels::cublasTnrm2_kernel<
        __half, float, 256>(
            const int n,
            const __half* RESTRICT x,
            const int incx,
            float* RESTRICT result);
#endif

// cublasTnrm2 kernel (__nv_bfloat16)
#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template
    __global__ void cublas_impl_kernels::cublasTnrm2_kernel<
        __nv_bfloat16, float, 256>(
            const int n,
            const __nv_bfloat16* RESTRICT x,
            const int incx,
            float* RESTRICT result);
#endif

// cublasTnrm2 kernel (float)
#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    __global__ void cublas_impl_kernels::cublasTnrm2_kernel<
        float, float, 256>(
            const int n,
            const float* RESTRICT x,
            const int incx,
            float* RESTRICT result);
#endif
#endif

// cublasTnrm2 kernel (double)
#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    __global__ void cublas_impl_kernels::cublasTnrm2_kernel<
        double, double, 256>(
            const int n,
            const double* RESTRICT x,
            const int incx,
            double* RESTRICT result);
#endif
#endif

// cublasTscal kernel (__half)
#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template
    __global__ void cublas_impl_kernels::cublasTscal_kernel<__half>(
            const int n,
            const __half alpha,
            __half* RESTRICT x,
            const int incx);
#endif

// cublasTscal kernel (__nv_bfloat16)
#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template
    __global__ void cublas_impl_kernels::cublasTscal_kernel<__nv_bfloat16>(
            const int n,
            const __nv_bfloat16 alpha,
            __nv_bfloat16* RESTRICT x,
            const int incx);
#endif

// cublasTscal kernel (float)
#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    __global__ void cublas_impl_kernels::cublasTscal_kernel<float>(
            const int n,
            const float alpha,
            float* RESTRICT x,
            const int incx);
#endif
#endif

// cublasTscal kernel (double)
#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
#if !defined(USE_CUBLAS) || (USE_CUBLAS != 1)
    template
    __global__ void cublas_impl_kernels::cublasTscal_kernel<double>(
            const int n,
            const double alpha,
            double* RESTRICT x,
            const int incx);
#endif
#endif
