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
cMatrix<DataType>::cMatrix()
{
}


// ==========
// destructor
// ==========

template <typename DataType>
cMatrix<DataType>::~cMatrix()
{
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
    assert((false) && "This function should no be called within this class");

    // Mark unused variables to avoid compiler warnings (-Wno-unused-parameter)
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
