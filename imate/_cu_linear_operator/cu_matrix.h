/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CU_LINEAR_OPERATOR_CU_MATRIX_H_
#define _CU_LINEAR_OPERATOR_CU_MATRIX_H_


// =======
// Headers
// =======

#include "../_definitions/types.h"  // FlagType, IndexType, LongIndexType
#include "./cu_linear_operator.h"  // cuLinearOperator


// ========
// c Matrix
// ========

/// \class   cuMatrix
///
/// \brief   Base class for constant matrices.
///
/// \details The prefix \c c in this class's name (and its derivatves), stands
///          for the \c cpp code, intrast to the \c cu prefix, which stands for
///          the cuda code. Most derived classes have a cuda counterpart.
///
/// \sa      cuAffineMatrixFunction


template <typename DataType>
class cuMatrix : public cuLinearOperator<DataType>
{
    public:

        // Member methods
        cuMatrix();
        explicit cuMatrix(int num_gpu_devices_);

        virtual ~cuMatrix();

        virtual void copy_host_to_device() = 0;
};


#endif  // _CU_LINEAR_OPERATOR_CU_MATRIX_H_
