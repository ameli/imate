/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CU_LINEAR_OPERATOR_CU_LINEAR_OPERATOR_H_
#define _CU_LINEAR_OPERATOR_CU_LINEAR_OPERATOR_H_

// =======
// Headers
// =======

#include <cusparse.h>  // cusparseHandle_t
#include "../_definitions/types.h"  // FlagType, IndexType, LongIndexType
                                    //
// cLinearOperatorBase
#include "../_c_linear_operator/c_linear_operator_base.h"

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
    #include <cublas_v2.h>  // cublasHandle_t, cublasCreate, cublasSetMathMode
#endif


// ==================
// cu Linear Operator
// ==================

/// \class   cuLinearOperator
///
/// \brief   Base class for linear operators. This class serves as interface
///          for all derived classes.
///
/// \details The prefix \c c in this class's name (and its derivatves), stands
///          for the \c cpp code, intrast to the \c cu prefix, which stands for
///          the cuda code. Most derived classes have a cuda counterpart.
///
/// \sa      cuMatrix,
///          cuAffineMatrixFunction,
///          cLinearOperator
///          cLinearOperatorBase

template <typename DataType>
class cuLinearOperator: virtual public cLinearOperatorBase
{
    public:

        // Member methods
        cuLinearOperator();

        explicit cuLinearOperator(const int num_gpu_devices_);

        virtual ~cuLinearOperator();

        cublasHandle_t get_cublas_handle() const;
        
        void set_parameters(DataType* parameters_);
        
        virtual DataType get_eigenvalue(
                const DataType* known_parameters,
                const DataType known_eigenvalue,
                const DataType* inquiry_parameters) const = 0;

        virtual void dot(
                const DataType* vector,
                DataType* product) = 0;

        virtual void transpose_dot(
                const DataType* vector,
                DataType* product) = 0;

    protected:

        // Member methods
        int query_gpu_devices() const;
        void initialize_cublas_handle();
        void initialize_cusparse_handle();

        // Member data
        int num_gpu_devices;
        bool copied_host_to_device;
        cublasHandle_t* cublas_handle;
        cusparseHandle_t* cusparse_handle;
        DataType* parameters;
};

#endif  // _CU_LINEAR_OPERATOR_CU_LINEAR_OPERATOR_H_
