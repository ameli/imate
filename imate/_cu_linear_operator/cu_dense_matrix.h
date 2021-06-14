/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CU_LINEAR_OPERATOR_CU_DENSE_MATRIX_H_
#define _CU_LINEAR_OPERATOR_CU_DENSE_MATRIX_H_


// =======
// Headers
// =======

#include "../_definitions/types.h"  // FlagType, LongIndexType
#include "../_c_linear_operator/c_dense_matrix.h"  // cDenseMatrix
#include "./cu_matrix.h"  // cuMatrix


// ===============
// cu Dense Matrix
// ===============

template <typename DataType>
class cuDenseMatrix :
    public cuMatrix<DataType>,
    public cDenseMatrix<DataType>
{
    public:

        // Member methods
        cuDenseMatrix();

        cuDenseMatrix(
                const DataType* A_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_,
                const FlagType A_is_row_major_,
                const int num_gpu_devices_);

        virtual ~cuDenseMatrix();

        virtual void dot(
                const DataType* device_vector,
                DataType* device_product);

        virtual void dot_plus(
                const DataType* device_vector,
                const DataType alpha,
                DataType* device_product);

        virtual void transpose_dot(
                const DataType* device_vector,
                DataType* device_product);

        virtual void transpose_dot_plus(
                const DataType* device_vector,
                const DataType alpha,
                DataType* device_product);

    protected:

        // Member methods
        virtual void copy_host_to_device();

        // Member data
        DataType** device_A;
};

#endif  // _CU_LINEAR_OPERATOR_CU_DENSE_MATRIX_H_
