/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_LINEAR_OPERATOR_C_CSR_MATRIX_H_
#define _C_LINEAR_OPERATOR_C_CSR_MATRIX_H_


// =======
// Headers
// =======

#include "../_definitions/types.h"  // FlagType, LongIndexType
#include "./c_matrix.h"  // cMatrix


// ============
// c CSR Matrix
// ============

/// \class   cCSRMatrix
///
/// \brief   Container for CSR matrices.
///
/// \details The \c cCSRMatrix holds a two-dimensional compressed sparse row
///          matrix, and can perofrom matrix-vector product and transposed
///          matrix-vector product.
///
/// \sa      cMatrix,
///          cDenseMatrix,
///          cCSCMatrix,
///          cCSRAffineMatrixFunction,
///          cuCSRMatrix

template <typename DataType>
class cCSRMatrix : public cMatrix<DataType>
{
    public:

        // Member methods
        cCSRMatrix();

        cCSRMatrix(
                const DataType* A_data_,
                const LongIndexType* A_indices_,
                const LongIndexType* A_index_pointer_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_,
                const FlagType A_is_symmetric);

        virtual ~cCSRMatrix();

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

#endif  // _C_LINEAR_OPERATOR_C_CSR_MATRIX_H_
