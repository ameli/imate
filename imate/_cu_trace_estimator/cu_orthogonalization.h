/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CU_TRACE_ESTIMATOR_CU_ORTHOGONALIZATION_H_
#define _CU_TRACE_ESTIMATOR_CU_ORTHOGONALIZATION_H_

// =======
// Imports
// =======

#include "../_definitions/types.h"  // IndexType, LongIndexType, FlagType

// Avoid CUBLAS numeration value not handled in switch [-Wswitch-enum] warning
#ifdef _MSC_VER
    #pragma warning(push, 0)  // Suppress all warnings from the followings
    #include <cublas_v2.h>  // cublasHandle_t
    #pragma warning(pop)  // Restore previous warning level
#elif defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
    #pragma warning(push, 0)
    #include <cublas_v2.h>  // cublasHandle_t
    #pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wswitch-enum"
    #include <cublas_v2.h>  // cublasHandle_t
    #pragma GCC diagnostic pop
#else
    #include <cublas_v2.h>  // cublasHandle_t
#endif


// ====================
// cu Orthogonalization
// ====================

/// \class cuOrthogonalization
///
/// \brief A static class for orthogonalization of vector bases. This class
///        acts as a templated namespace, where all member methods are *public*
///        and *static*.
///
/// \sa    RandomVectors

template <typename DataType>
class cuOrthogonalization
{
    public:

        // Gram-Schmidt Process
        static void gram_schmidt_process(
                cublasHandle_t cublas_handle,
                const DataType* V,
                const LongIndexType vector_size,
                const IndexType num_vectors,
                const IndexType last_vector,
                const FlagType num_ortho,
                DataType* r);

        // Orthogonalize Vectors
        static void orthogonalize_vectors(
                cublasHandle_t cublas_handle,
                DataType* vectors,
                const LongIndexType vector_size,
                const IndexType num_vectors);
};

#endif  // _CU_TRACE_ESTIMATOR_CU_ORTHOGONALIZATION_H_
