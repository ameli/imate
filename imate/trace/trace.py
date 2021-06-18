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

from ._exact_method import exact_method
from ._eigenvalue_method import eigenvalue_method
from ._slq_method import slq_method


# =====
# trace
# =====

def trace(A, method='slq', **options):
    """
    Computes the trace of a matrix.
    See :ref:`Compute Trace User Guide<trace_UserGuide>` for details.

    :param A: A square matrix
    :type A: numpy.ndarray

    :param method: One of ``'cholesky'``, ``'hutchinson'``, or ``'slq'``.
    :type method: string

    :param options: Options for either of the methods.
    :type options: ``**kwargs``

    :return: trace of matrix
    :rtype: float

    :raises RunTimeError: Method is not recognized.

    **Methods:**

    The trace is computed with one of these three methods in this
    function.

    ================  =============================  =============
    ``'method'``       Description                    Type
    ================  =============================  =============
    ``exact``         Direct method                  exact
    ``eigenvalue``    Eigenvalue method              exact
    ``'slq'``         Stochastic Lanczos Quadrature  approximation
    ================  =============================  =============

    Depending the method, this function calls these modules:

    * :mod:`imate.trace.slq_method`

    **Examples:**

    .. code-block:: python

       >>> from imate import generate_matrix
       >>> from imate import trace

       >>> # Generate a symmetric positive-definite matrix of size 20**2
       >>> A = generate_matrix(size=20, dimension=2)

       >>> # Compute trace
       >>> trace = trace(A)

    The above example uses the *slq* method by default.
    In the next example, we apply the *Hutchinson's randomized estimator*
    method.

    .. code-block:: python

       >>> trace = trace(A, method='slq', num_samples=20)

    Using the stochastic Lanczos quadrature method with Lanczos
    tri-diagonalization (this method is suitable for symmetric matrices)

    .. code-block:: python

       >>> trace = trace(A, method='slq', num_samples=20, lanczos_degree=30)

    Using the stochastic Lanczos quadrature method with Golub-Kahn
    bi-diagonalization:

    .. code-block:: python

       >>> trace = trace(A, method='slq', num_samples=20, lanczos_degree=30,
       ...               symmetric=False)

    The above method is suitable for non-symmetric matrices, despite the matrix
    in the above examples is symmetric.
    """

    if method == 'exact':
        trace, info = exact_method(A, **options)

    elif method == 'eigenvalue':
        trace, info = eigenvalue_method(A, **options)

    elif method == 'slq':
        trace, info = slq_method(A, **options)

    else:
        raise RuntimeError('Method "%s" is not recognized.' % method)

    return trace, info
