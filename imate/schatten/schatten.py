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

from ..logdet import logdet
from ..trace import trace
from ..traceinv import traceinv
import numpy


# ========
# schatten
# ========

def schatten(
        A,
        gram=False,
        p=0,
        method='cholesky',
        **options):
    """
    Computes the log-determinant of full-rank matrix.

    Parameters
    ----------

        A : numpy.ndarray, scipy.sparse matrix
            An invertible matrix.

        options : `**kwargs`
            Options for either of the methods.

    Return
    ------
        logdet : float
            Log-determinant of `A`

    Notes
    -----

        For computing the *trace of inverse* with the stochastic Lanczos
        quadrature method (see :mod:`imate.traceinv.slq_method`), the
        preferred algorithm is the Lanczos tri-diagonalization, as opposed to
        Golub-Kahn bi-diagonalization.

        In contrast to the above, the preferred slq method for computing
        *log-determinant* is the Golub-Kahn bi-diagonalization. The reason is
        that if the matrix :math:`\\mathbf{A}` has many singular values close
        to zero, bi-diagonalization performs better, and this matters when we
        compute determinant.

    References
    ----------

        * Ubaru, S., Chen, J., and Saad, Y. (2017).
          Fast estimation of :math:`\\mathrm{tr}(f(A))` via stochastic Lanczos
          quadrature. *SIAM Journal on Matrix Analysis and Applications*,
          38(4), 1075-1099.
          `doi: 10.1137/16M1104974 <https://doi.org/10.1137/16M1104974>`_

    Examples
    --------

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

    if not isinstance(p, (int, numpy.integer, float)):
        raise ValueError('"p" should be float or integer.')

    n = A.shape[0]

    if p == 0:
        # Note, p in logdet is different than p in schatten.
        logdet_, info = logdet(A, gram=gram, p=1.0, method=method, **options)
        schatten = numpy.exp(logdet_ / n)
    elif p > 0:
        trace_, info = trace(A, gram=gram, p=p, method=method, **options)
        schatten = (trace_ / n)**(1.0/p)
    elif p < 0:
        traceinv_, info = traceinv(A, gram=gram, p=numpy.abs(p), method=method,
                                   **options)
        schatten = (traceinv_ / n)**(1.0/p)

    return schatten, info
