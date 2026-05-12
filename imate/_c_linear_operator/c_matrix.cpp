/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


// =======
// Headers
// =======

#include "./c_matrix.h"
#include <cassert>  // assert


// =============
// constructor 1
// =============

/// \brief Default constructor.
///

template <typename DataType>
cMatrix<DataType>::cMatrix():

    // Initializer list
    A_is_symmetric(0)
{
}


// =============
// constructor 2
// =============

/// \brief     Constructor.
///
/// \param[in] A_is_symmetric_
///            If \c 1, it is assumed that the matrix is symmetric, otherwise
///            set to \c 0.

template <typename DataType>
cMatrix<DataType>::cMatrix(const FlagType A_is_symmetric_):

    // Initializer list
    A_is_symmetric(A_is_symmetric_)
{
}


// ==========
// destructor
// ==========

/// \brief Destructor
///

template <typename DataType>
cMatrix<DataType>::~cMatrix()
{
}


// ============
// set symmetry
// ============

/// \brief     Specify whether the matrix is symmetic or non-symmetric.
///
/// \details   This function overwrites the symmetry status that has been set
///            by the constructor.
///
/// \param[in] symmetric
///            Boolean. If set to \c 1, the matrix is assumed to be symmetric.
///            Otherwiese non-symmetric.

template <typename DataType>
void cMatrix<DataType>::set_symmetry(const FlagType symmetric)
{
    if (symmetric == 1)
    {
        this->A_is_symmetric = 1;
    }
    else
    {
        this->A_is_symmetric = 0;
    }
}


// ==============
// get eigenvalue
// ==============

/// \brief     This virtual function is implemented from its pure virtual
///            function of the base class. In this class, this functio has no
///            use and was only implemented so that this class be able to
///            be instantiated (due to the pure virtual function).
///
/// \param[in] known_parameters
///            A set of parameters of the operator where the corresponding
///            eigenvalue of the parameter is known for.
/// \param[in] known_eigenvalue
///            The known eigenvalue of the operator for the known parameters.
/// \param[in] inquiry_parameters
///            A set of inquiry parameters of the operator where the
///            corresponding eigenvalue of the operator is sought.
/// \return    The eigenvalue of the operator corresponding the inquiry
///            parameters.

template <typename DataType>
DataType cMatrix<DataType>::get_eigenvalue(
        const DataType* known_parameters,
        const DataType known_eigenvalue,
        const DataType* inquiry_parameters) const
{
    assert((false) && "This function should not be called within this class");

    // Void unused variables to avoid compiler warnings (-Wno-unused-parameter)
    (void) known_parameters;
    (void) known_eigenvalue;
    (void) inquiry_parameters;

    return 0;
}


// ===============================
// Explicit template instantiation
// ===============================

template class cMatrix<float>;
template class cMatrix<double>;
template class cMatrix<long double>;
