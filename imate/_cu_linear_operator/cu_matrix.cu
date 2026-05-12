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

#include "./cu_matrix.h"
#include <cassert>  // assert
#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __nv_bfloat16


// =============
// constructor 1
// =============

/// \brief Default constructor.
///

template <typename DataType>
cuMatrix<DataType>::cuMatrix():
    
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
cuMatrix<DataType>::cuMatrix(const FlagType A_is_symmetric_):

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
cuMatrix<DataType>::~cuMatrix()
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
void cuMatrix<DataType>::set_symmetry(const FlagType symmetric)
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
DataType cuMatrix<DataType>::get_eigenvalue(
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

#if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template class cuMatrix<__nv_fp8_e5m2>;
#endif

#if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template class cuMatrix<__nv_fp8_e4m3>;
#endif

#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template class cuMatrix<__half>;
#endif

#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template class cuMatrix<__nv_bfloat16>;
#endif

#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template class cuMatrix<float>;
#endif

#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template class cuMatrix<double>;
#endif
