/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_TRACE_ESTIMATOR_DIAGONALIZATION_H_
#define _C_TRACE_ESTIMATOR_DIAGONALIZATION_H_

// =======
// Headers
// =======

#include "../_definitions/types.h"  // IndexType


// ===============
// Diagonalization
// ===============

/// \class Diagonalization
///
/// \brief A static class to find eigenvalues and eigenvectors (diagonalize)
///        tridiagonal and bidiagonal matrices. This class acts as a templated
///        namespace, where all member methods are *public* and *static*.
///
/// \sa    StochasticLanczosQuadrature

template <typename DataType>
class Diagonalization
{
    public:

        // eigh tridiagonal
        static int eigh_tridiagonal(
                DataType* diagonals,
                DataType* subdiagonals,
                DataType* eigenvectors,
                IndexType matrix_size);

        // svd bidiagonal
        static int svd_bidiagonal(
                DataType* diagonals,
                DataType* subdiagonals,
                DataType* U,
                DataType* Vt,
                IndexType matrix_size);
};

#endif  // _C_TRACE_ESTIMATOR_DIAGONALIZATION_H_
