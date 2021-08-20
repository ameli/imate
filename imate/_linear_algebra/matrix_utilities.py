# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

from scipy.sparse import isspmatrix


# ==================
# get data type name
# ==================

def get_data_type_name(A):
    """
    Returns the dtype as string.
    """

    if A.dtype == 'float32':
        data_type_name = b'float32'

    elif A.dtype == 'float64':
        data_type_name = b'float64'

    elif A.dtype == 'float128':
        data_type_name = b'float128'

    else:
        raise TypeError('Data type should be either "float32" or "float64"')

    return data_type_name


# =======
# get nnz
# =======

def get_nnz(A):
    """
    Returns the number of non-zero elements of a matrix.
    """

    if isspmatrix(A):
        return A.nnz
    else:
        return A.shape[0] * A.shape[1]


# ===========
# get density
# ===========

def get_density(A):
    """
    Returns the density of non-zero elements of a matrix.
    """

    if isspmatrix(A):
        return get_nnz(A) / (A.shape[0] * A.shape[1])
    else:
        return 1.0
