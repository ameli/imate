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


// =========
// cu Matrix
// =========

/// \class   cuMatrix
///
/// \brief   Base class for constant matrices.
///
/// \details The prefix \c c in this class's name (and its derivatves), stands
///          for the \c cpp code, intrast to the \c cu prefix, which stands for
///          the cuda code. Most derived classes have a cuda counterpart.
///
/// \sa      cuLinearOperator,
///          cuAffineMatrixFunction,
///          cuDenseMatrix,
///          cuCSRMatrix,
///          cuCSCMatrix,
///          cMatrix

template <typename DataType>
class cuMatrix : virtual public cuLinearOperator<DataType>
{
    public:

        // Member methods
        cuMatrix();
        
        explicit cuMatrix(const FlagType A_is_symmetric_);

        virtual ~cuMatrix();

        virtual void copy_host_to_device() = 0;
        
        DataType get_eigenvalue(
                const DataType* known_parameters,
                const DataType known_eigenvalue,
                const DataType* inquiry_parameters) const;

        virtual FlagType is_identity_matrix() const = 0;

        virtual void set_symmetry(const FlagType symmetric);

        virtual void dot_plus(
                const DataType* vector,
                const DataType alpha,
                DataType* product) = 0;

        virtual void transpose_dot_plus(
                const DataType* vector,
                const DataType alpha,
                DataType* product) = 0;

    protected:

        // Member data
        FlagType A_is_symmetric;
};


#endif  // _CU_LINEAR_OPERATOR_CU_MATRIX_H_
