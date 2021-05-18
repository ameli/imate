/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_LINEAR_OPERATOR_C_CSC_MATRIX_H_
#define _C_LINEAR_OPERATOR_C_CSC_MATRIX_H_


// =======
// Headers
// =======

#include "../_definitions/types.h"  // LongIndexType, FlagType
#include "./c_matrix.h"  // cMatrix


// ============
// c CSC Matrix
// ============

template <typename DataType>
class cCSCMatrix : public cMatrix<DataType>
{
    public:

        // Member methods
        cCSCMatrix();

        cCSCMatrix(
                const DataType* A_data_,
                const LongIndexType* A_indices_,
                const LongIndexType* A_index_pointer_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_);

        virtual ~cCSCMatrix();

        virtual FlagType is_identity_matrix() const;

        LongIndexType get_nnz() const;

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
        const DataType* A_data;
        const LongIndexType* A_indices;
        const LongIndexType* A_index_pointer;
};

#endif  // _C_LINEAR_OPERATOR_C_CSC_MATRIX_H_
