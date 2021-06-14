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

#include "./cu_affine_matrix_function.h"
#include <cassert>  // assert
#include "../_definitions/debugging.h"  // ASSERT
#include "../_cu_basic_algebra/cu_vector_operations.h"  // cuVectorOperations
#include "../_cuda_utilities/cuda_interface.h"  // CudaInterface


// ===========
// constructor
// ===========

/// \brief Constructor.
///

template <typename DataType>
cuAffineMatrixFunction<DataType>::cuAffineMatrixFunction():
    B_is_identity(false)
{
    // This class has one parameter that is t in A+tB
    this->num_parameters = 1;
}


// ==========
// destructor
// ==========

/// \brief Virtual destructor.

template <typename DataType>
cuAffineMatrixFunction<DataType>::~cuAffineMatrixFunction()
{
}


// ==============
// get eigenvalue
// ==============

/// \brief     This function defines an analytic relationship between a given
///            set of parameters and the corresponding eigenvalue of the
///            operator. Namely, given a set of parameters and a *known*
///            eigenvalue of the operator for that specific set of parameters,
///            this function obtains the eigenvalue of the operator for an
///            other given set of parameters.
///
/// \details   A relation between eigenvalue(s) and the set of parameters can
///            be made when the matrix :math:`\\mathbf{B}` is equal to the
///            identity matrix \f$ \mathbf{I} \f$, and corresponding linear
///            operator is as follows:
///
///            \f[
///                \mathbf{A}(t) = \mathbf{A} + t \mathbf{I}
///            \f]
///
///            Then, the eigenvalues \f$ \lambda \f$ of the operator is
///
///            \f[
///                \lambda(\mathbf{A}(t)) = \lambda(\mathbf{A}) + t.
///            \f]
///
///            Thus, by knowing the eigenvalue \f$ \lambda_{t_0} \f$ at the
///            known parameter \f$ t_0 \f$, the eigenvalue \f$ \lambda_{t} \f$
///            at the inquiry parameter \f$ t \f$ can be obtained.
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
DataType cuAffineMatrixFunction<DataType>::get_eigenvalue(
        const DataType* known_parameters,
        const DataType known_eigenvalue,
        const DataType* inquiry_parameters) const
{
    ASSERT((this->eigenvalue_relation_known == 1),
            "An eigenvalue relation is not known. This function should be "
            "called only when the matrix B is a scalar multiple of the "
            "identity matrix");

    // Shift the eigenvalue by the parameter
    DataType inquiry_eigenvalue = \
        known_eigenvalue - known_parameters[0] + inquiry_parameters[0];

    return inquiry_eigenvalue;
}


// =================
// add scaled vector
// =================

/// \brief         Performs the operation \f$ \boldsymbol{c} = \boldsymbol{c} +
///                \alpha * \boldsymbol{b} \f$, where \f$ \boldsymbol{b} \f$ is
///                an input vector scaled by \f$ \alpha \f$ and \f$
///                \boldsymbol{c} \f$ it the output vector.
///
/// \param[in]     input_vector
///                The input 1D vector of size \c vector_size.
/// \param[in]     vector_size
///                The size of both \c input_vector and \c output_vector
/// \param[in]     scale
///                The scalar scale at whch the input vector is multiplied by.
/// \param[in,out] output_vector
///                The output 1D vector of size \c vector_size. This array will
///                be written in-place.

template <typename DataType>
void cuAffineMatrixFunction<DataType>::_add_scaled_vector(
        const DataType* input_vector,
        const LongIndexType vector_size,
        const DataType scale,
        DataType* output_vector) const
{
    // Get device id
    int device_id = CudaInterface<DataType>::get_device();

    // Subtracting two vectors with minus scale sign, which is adding.
    cuVectorOperations<DataType>::subtract_scaled_vector(
            this->cublas_handle[device_id], input_vector, vector_size, -scale,
            output_vector);
}


// ===============================
// Explicit template instantiation
// ===============================

template class cuAffineMatrixFunction<float>;
template class cuAffineMatrixFunction<double>;
