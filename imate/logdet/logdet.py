# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

from ._eigenvalue_method import eigenvalue_method
from ._cholesky_method import cholesky_method
from ._slq_method import slq_method


# ======
# logdet
# ======

def logdet(A, method='cholesky', **options):
    """
    Computes the log-determinant of full-rank matrix ``A``.

    :param A: A full-rank matrix.
    :type A: numpy.ndarray or scipy.sparse.csc_matrix

    :param options: Options for either of the methods.
    :type options: ``**kwargs``

    :return: Log-determinant of ``A``
    :rtype: float

    .. note::

        For computing the *trace of inverse* with the stochastic Lanczos
        quadrature method (see :mod:`imate.traceinv.slq_method`), the
        preferred algorithm is the Lanczos tri-diagonalization, as opposed to
        Golub-Kahn bi-diagonalization.

        In contrast to the above, the preferred slq method for computing
        *log-determinant* is the Golub-Kahn bi-diagonalization. The reason is
        that if the matrix :math:`\\mathbf{A}` has many singular values close
        to zero, bi-diagonalization performs better, and this matters when we
        compute determinant.

    References:
        * Ubaru, S., Chen, J., and Saad, Y. (2017).
          Fast estimation of :math:`\\mathrm{tr}(f(A))` via stochastic Lanczos
          quadrature. *SIAM Journal on Matrix Analysis and Applications*,
          38(4), 1075-1099.
          `doi: 10.1137/16M1104974 <https://doi.org/10.1137/16M1104974>`_

    **Examples:**

    .. code-block:: python

        >>> # Import packages
        >>> from imate import logdet
        >>> from imate import generate_matrix

        >>> # Generate a sample matrix
        >>> A = generate_matrix(size=20, dimension=2)

        >>> # Compute log-determinant with Cholesky method (default method)
        >>> logdet_1 = logdet(A)

        >>> # Compute log-determinant with stochastic Lanczos quadrature method
        >>> # using Lanczos tridiagonalization
        >>> logdet_2 = logdet(A, method='slq', num_iterations=20,
        ...     lanczosdegree=20, symmetric=True)

        >>> # Compute log-determinant with stochastic Lanczos quadrature method
        >>> # using Golub-Khan bi-diagonalization
        >>> logdet_3 = logdet(A, method='slq', num_iterations=20,
        ...     lanczos_degree=20, symmetric=False)
    """

    if method == 'eigenvalue':
        trace, info = eigenvalue_method(A, **options)

    elif method == 'cholesky':
        trace, info = cholesky_method(A, **options)

    elif method == 'slq':
        trace, info = slq_method(A, **options)

    else:
        raise ValueError('Method "%s" is invalid.' % (method))

    return trace, info
