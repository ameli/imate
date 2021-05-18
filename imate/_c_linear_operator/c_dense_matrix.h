/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_LINEAR_OPERATOR_C_DENSE_MATRIX_H_
#define _C_LINEAR_OPERATOR_C_DENSE_MATRIX_H_


// =======
// Headers
// =======

#include "../_definitions/types.h"  // FlagType, LongIndexType
#include "./c_matrix.h"  // cMatrix


// ==============
// c Dense Matrix
// ==============

template <typename DataType>
class cDenseMatrix : public cMatrix<DataType>
{
    public:

        // Member methods
        cDenseMatrix();

        cDenseMatrix(
                const DataType* A_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_,
                const FlagType A_is_row_major_);

        virtual ~cDenseMatrix();

        virtual FlagType is_identity_matrix() const;

        virtual void dot(
                const DataType* vector,
                DataType* product);

        virtual void dot_plus(
                const DataType* vector,
                const DataType alpha,
                DataType* product);

        virtual void transpose_dot(
                const DataType* vector,
                DataType* product);

        virtual void transpose_dot_plus(
                const DataType* vector,
                const DataType alpha,
                DataType* product);

    protected:

        // Member data
        const DataType* A;
        const FlagType A_is_row_major;
};

#endif  // _C_LINEAR_OPERATOR_C_DENSE_MATRIX_H_
