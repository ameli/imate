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

from ._eigenvalue_method import eigenvalue_method
from ._cholesky_method import cholesky_method
from ._hutchinson_method import hutchinson_method
from ._slq_method import slq_method


# ========
# traceinv
# ========

def traceinv(A, method='slq', **options):
    """
    Computes the trace of inverse of a matrix.
    See :ref:`Compute Trace of Inverse User Guide<traceinv_UserGuide>` for
    details.

    :param A: Invertible matrix
    :type A: numpy.ndarray

    :param method: One of ``'cholesky'``, ``'hutchinson'``, or ``'slq'``.
    :type method: string

    :param options: Options for either of the methods.
    :type options: ``**kwargs``

    :return: trace of inverse of matrix
    :rtype: float

    :raises RunTimeError: Method is not recognized.

    **Methods:**

    The trace of inverse is computed with one of these three methods in this
    function.

    ================  =============================  =============
    ``'method'``       Description                    Type
    ================  =============================  =============
    ``eigenvalue``    Eigenvalue method              exact
    ``'cholesky'``    Cholesky method                exact
    ``'hutchinson'``  Hutchinson method              approximation
    ``'slq'``         Stochastic Lanczos Quadrature  approximation
    ================  =============================  =============

    Depending the method, this function calls these modules:

    * :mod:`imate.traceinv.cholesky_method`
    * :mod:`imate.traceinv.hutchinson_method`
    * :mod:`imate.traceinv.slq_method`

    **Examples:**

    .. code-block:: python

       >>> from imate import generate_matrix
       >>> from imate import traceinv

       >>> # Generate a symmetric positive-definite matrix of size 20**2
       >>> A = generate_matrix(size=20, dimension=2)

       >>> # Compute trace of inverse
       >>> trace = traceinv(A)

    The above example uses the *slq* method by default.
    In the next example, we apply the *Hutchinson's randomized estimator*
    method.

    .. code-block:: python

       >>> trace = traceinv(A, method='hutchinson', num_samples=20)

    Using the stochastic Lanczos quadrature method with Lanczos
    tri-diagonalization (this method is suitable for symmetric matrices)

    .. code-block:: python

       >>> trace = traceinv(A, method='slq', num_samples=20,
       ...     lanczos_degree=30)

    Using the stochastic Lanczos quadrature method with Golub-Kahn
    bi-diagonalization:

    .. code-block:: python

       >>> trace = traceinv(A, method='slq', num_samples=20,
       ...     lanczos_degree=30, symmetric=False)

    The above method is suitable for non-symmetric matrices, despite the matrix
    in the above examples is symmetric.
    """

    if method == 'eigenvalue':
        trace, info = eigenvalue_method(A, **options)

    elif method == 'cholesky':
        trace, info = cholesky_method(A, **options)

    elif method == 'hutchinson':
        trace, info = hutchinson_method(A, **options)

    elif method == 'slq':
        trace, info = slq_method(A, **options)

    else:
        raise RuntimeError('Method "%s" is not recognized.' % method)

    return trace, info
