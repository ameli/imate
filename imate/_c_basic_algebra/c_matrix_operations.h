/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_BASIC_ALGEBRA_C_MATRIX_OPERATIONS_H_
#define _C_BASIC_ALGEBRA_C_MATRIX_OPERATIONS_H_

// =======
// Headers
// =======

#include "../_definitions/types.h"  // IndexType, LongIndexType, FlagType


// ===================
// c Matrix Operations
// ===================

/// \class cMatrixOperations
///
/// \brief   A static class for matrix-vector operations, which are similar to
///          the level-2 operations of the BLAS library. This class acts as a
///          templated namespace, where all member methods are *public* and
///          *static*.
///
/// \details This class implements matrix-ector multiplication for three types
///          of matrices:
///
///          * Dense matrix (both row major and column major)
///          * Compressed sparse row matrix (CSR)
///          * Compressed sparse column matrix (CSC)
///
///          For each of the above matrix types, there are four kinds of matrix
///          vector multiplications implemented.
///
///          1. \c dot : performs \f$ \boldsymbol{c} = \mathbf{A}
///             \boldsymbol{b} \f$.
///          2. \c dot_plus : performs \f$ \boldsymbol{c} = \boldsymbol{c} +
///             \alpha \mathbf{A} \boldsymbol{b} \f$.
///          3. \c transpose_dot : performs \f$ \boldsymbol{c} =
///             \mathbf{A}^{\intercal} \boldsymbol{b} \f$.
///          4. \c transpose_dot_plus : performs \f$ \boldsymbol{c} =
///             \boldsymbol{c} + \alpha \mathbf{A}^{\intercal} \boldsymbol{b}
///             \f$.
///
/// \sa      cVectorOperations

template <typename DataType>
class cMatrixOperations
{
    public:

        // dense matvec
        static void dense_matvec(
                const DataType* A,
                const DataType* b,
                const LongIndexType num_rows,
                const LongIndexType num_columns,
                const FlagType A_is_row_major,
                DataType* c);

        // dense matvec plus
        static void dense_matvec_plus(
                const DataType* A,
                const DataType* b,
                const DataType alpha,
                const LongIndexType num_rows,
                const LongIndexType num_columns,
                const FlagType A_is_row_major,
                DataType* c);

        // dense transposed matvec
        static void dense_transposed_matvec(
                const DataType* A,
                const DataType* b,
                const LongIndexType num_rows,
                const LongIndexType num_columns,
                const FlagType A_is_row_major,
                DataType* c);

        // dense transposed matvec plus
        static void dense_transposed_matvec_plus(
                const DataType* A,
                const DataType* b,
                const DataType alpha,
                const LongIndexType num_rows,
                const LongIndexType num_columns,
                const FlagType A_is_row_major,
                DataType* c);

        // CSR matvec
        static void csr_matvec(
                const DataType* A_data,
                const LongIndexType* A_column_indices,
                const LongIndexType* A_index_pointer,
                const DataType* b,
                const LongIndexType num_rows,
                DataType* c);

        // CSR matvec plus
        static void csr_matvec_plus(
                const DataType* A_data,
                const LongIndexType* A_column_indices,
                const LongIndexType* A_index_pointer,
                const DataType* b,
                const DataType alpha,
                const LongIndexType num_rows,
                DataType* c);

        // CSR transposed matvec
        static void csr_transposed_matvec(
                const DataType* A_data,
                const LongIndexType* A_column_indices,
                const LongIndexType* A_index_pointer,
                const DataType* b,
                const LongIndexType num_rows,
                const LongIndexType num_columns,
                DataType* c);

        // CSR transposed matvec plus
        static void csr_transposed_matvec_plus(
                const DataType* A_data,
                const LongIndexType* A_column_indices,
                const LongIndexType* A_index_pointer,
                const DataType* b,
                const DataType alpha,
                const LongIndexType num_rows,
                DataType* c);

        // CSC matvec
        static void csc_matvec(
                const DataType* A_data,
                const LongIndexType* A_row_indices,
                const LongIndexType* A_index_pointer,
                const DataType* b,
                const LongIndexType num_rows,
                const LongIndexType num_columns,
                DataType* c);

        // CSC matvec plus
        static void csc_matvec_plus(
                const DataType* A_data,
                const LongIndexType* A_row_indices,
                const LongIndexType* A_index_pointer,
                const DataType* b,
                const DataType alpha,
                const LongIndexType num_columns,
                DataType* c);

        // CSC transposed matvec
        static void csc_transposed_matvec(
                const DataType* A_data,
                const LongIndexType* A_row_indices,
                const LongIndexType* A_index_pointer,
                const DataType* b,
                const LongIndexType num_columns,
                DataType* c);

        // CSC transposed matvec plus
        static void csc_transposed_matvec_plus(
                const DataType* A_data,
                const LongIndexType* A_row_indices,
                const LongIndexType* A_index_pointer,
                const DataType* b,
                const DataType alpha,
                const LongIndexType num_columns,
                DataType* c);

        // Create Band Matrix
        static void create_band_matrix(
                const DataType* diagonals,
                const DataType* supdiagonals,
                const IndexType non_zero_size,
                const FlagType tridiagonal,
                DataType** matrix);
};

#endif  // _C_BASIC_ALGEBRA_C_MATRIX_OPERATIONS_H_
