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


// =============
// constructor 1
// =============

template <typename DataType>
cuMatrix<DataType>::cuMatrix()
{
}


// =============
// constructor 2
// =============

template <typename DataType>
cuMatrix<DataType>::cuMatrix(int num_gpu_devices_):
    cuLinearOperator<DataType>(num_gpu_devices_)
{
}


// ==========
// destructor
// ==========

template <typename DataType>
cuMatrix<DataType>::~cuMatrix()
{
}


// ===============================
// Explicit template instantiation
// ===============================

template class cuMatrix<float>;
template class cuMatrix<double>;
