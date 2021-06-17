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

#include <cublas_v2.h>  // cublasHandle_t
#include <cusparse.h>  // cusparseHandle_t
#include "../_definitions/types.h"  // FlagType, IndexType, LongIndexType
#include "../_c_linear_operator/c_linear_operator.h"  // cLinearOperator


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

template <typename DataType>
class cuLinearOperator: virtual public cLinearOperator<DataType>
{
    public:

        // Member methods
        cuLinearOperator();
        explicit cuLinearOperator(int num_gpu_devices_);

        virtual ~cuLinearOperator();

        cublasHandle_t get_cublas_handle() const;

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
};

#endif  // _CU_LINEAR_OPERATOR_CU_LINEAR_OPERATOR_H_
