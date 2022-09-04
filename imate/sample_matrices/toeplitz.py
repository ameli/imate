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

import numpy
from scipy.sparse import coo_matrix

__all__ = ['toeplitz', 'toeplitz_trace', 'toeplitz_traceinv',
           'toeplitz_logdet', 'toeplitz_schatten']


# ========
# toeplitz
# ========

def toeplitz(
        a,
        b,
        size=20,
        gram=False,
        format='csr',
        dtype=r'float64'):
    """
    Generate a sparse Toeplitz matrix for test purposes.

    A bi-diagonal Toeplitz matrix is generated using `a` and `b` as

    .. math::
        :label: toeplitz-A

        \\mathbf{A} =
        \\begin{bmatrix}
            a & b & 0 &\\cdots & \\cdots & 0 \\

            0 & a & b & \\ddots && \\vdots \\

            0 & 0 & \\ddots & \\ddots & \\ddots & \\vdots \\

            \\vdots & \\ddots & \\ddots & \\ddots & b & 0 \\

            \\vdots & & \\ddots & 0 & a & b \\

            0 & \\cdots & \\cdots & 0 & 0 & a
        \\end{bmatrix}

    If ``gram=True``, the Gramian of the above matrix is generated, which is
    :math:`\\mathbf{B} = \\mathbf{A}^{\\intercal} \\mathbf{A}`, namely

    .. math::
        :label: toeplitz-B

        \\mathbf{B} =
        \\begin{bmatrix}
            a^2 & ab & 0 &\\cdots & \\cdots & 0 \\

            ab & a^2+b^2 & ab & \\ddots && \\vdots \\

            0 & ab & \\ddots & \\ddots & \\ddots & \\vdots \\

            \\vdots & \\ddots & \\ddots & \\ddots & b & 0 \\

            \\vdots & & \\ddots & ab & a^2+b^2 & ab \\

            0 & \\cdots & \\cdots & 0 & ab & a^2+b^2
        \\end{bmatrix}

    Parameters
    ----------

    a : float
        The diagonal elements of the Toeplitz matrix.

    b : float
        The upper off-diagonals element of the Toeplitz matrix.

    size : int, default=20
        Size of the square matrix.

    gram : bool, default=False
        If `False`, the bi-diagonal matrix :math:`\\mathbf{A}` in
        :math:numref:`toeplitz-A` is returned. If `True`, the Gramian
        tri-diagonal matrix :math:`\\mathbf{B}` in :math:numref:`toeplitz-B` is
        returned.

    format : {'csr', 'csc'}, default='csr'
        The format of the sparse matrix. `CSR` is the compressed sparse row and
        `CSC` is the compressed sparse column format.

    dtype : {'float32', 'float64', 'float128'}, default='float64'
        The data type of the matrix.

    Returns
    -------

    A : scipy.sparse.csr or scipy.sparse.csc, (n, n)
        Bi-diagonal or tri-diagonal (if ``grid=True``) Toeplitz matrix

    See Also
    --------

    imate.correlation_matrix
    imate.sample_matrices.toeplitz_logdet
    imate.sample_matrices.toeplitz_trace
    imate.sample_matrices.toeplitz_traceinv

    Notes
    -----

    The matrix functions of the Toeplitz matrix (such as log-determinant,
    trace of its inverse, etc) is known analytically. As such, this matrix can
    be used to test the accuracy of randomized algorithms for computing matrix
    functions.

    .. warning::

        All eigenvalues of the generated Toeplitz matrix are equal to
        :math:`a`. So, in applications where a matrix with distinct
        eigenvalues is needed, this matrix is not suitable. For such
        applications, use :func:`imate.correlation_matrix` instead.

    To generate a symmetric and positive-definite matrix, set ``gram`` to
    `True`.

    Examples
    --------

    Generate bi-diagonal matrix:

    .. code-block:: python

        >>> from imate.sample_matrices import toeplitz
        >>> a, b = 2, 3

        >>> # Bi-diagonal matrix
        >>> A = toeplitz(a, b, size=6, format='csr', dtype='float128')

        >>> print(A.dtype)
        dtype('float128')

        >>> print(type(A))
        scipy.sparse.csr.csr_matrix

        >>> # Convert sparse to dense numpy array to display the matrix
        >>> A.toarray()
        array([[2., 3., 0., 0., 0., 0.],
               [0., 2., 3., 0., 0., 0.],
               [0., 0., 2., 3., 0., 0.],
               [0., 0., 0., 2., 3., 0.],
               [0., 0., 0., 0., 2., 3.],
               [0., 0., 0., 0., 0., 2.]])

    Create a tri-diagonal Matrix:

    .. code-block:: python

        >>> # Tri-diagonal Gramian matrix
        >>> B = toeplitz(a, b, size=6, gram=True)
        >>> B.toarray()
        array([[ 4.,  6.,  0.,  0.,  0.,  0.],
               [ 6., 13.,  6.,  0.,  0.,  0.],
               [ 0.,  6., 13.,  6.,  0.,  0.],
               [ 0.,  0.,  6., 13.,  6.,  0.],
               [ 0.,  0.,  0.,  6., 13.,  6.],
               [ 0.,  0.,  0.,  0.,  6., 13.]])
    """

    _check_arguments(a, b, size, gram, format, dtype)

    data = numpy.empty((2*size-1, ), dtype=dtype)
    row = numpy.empty((2*size-1, ), dtype=int)
    col = numpy.empty((2*size-1, ), dtype=int)

    # Fill diagonals
    row[:size] = numpy.arange(size)
    col[:size] = numpy.arange(size)
    data[:size] = a

    # Fill off-diagonal
    row[size:] = numpy.arange(size-1)
    col[size:] = numpy.arange(1, size)
    data[size:] = b

    A = coo_matrix((data, (row, col)), shape=(size, size))

    if gram:
        A = A.T @ A

    if format == 'csc':
        A = A.tocsc()
    elif format == 'csr':
        A = A.tocsr()

    return A


# ===============
# check arguments
# ===============

def _check_arguments(
        a,
        b,
        size,
        gram,
        format,
        dtype):
    """
    Checks the type and values of the input arguments.
    """

    # Check a
    if a is None:
        raise ValueError('"a" cannot be None.')
    elif not isinstance(a, (int, numpy.integer, float)):
        raise TypeError('"a" should be integer or float.')

    # Check b
    if b is None:
        raise ValueError('"b" cannot be None.')
    elif not isinstance(b, (int, numpy.integer, float)):
        raise TypeError('"b" should be integer or float.')

    # Check size
    if size is None:
        raise TypeError('"size" cannot be None.')
    elif not numpy.isscalar(size):
        raise TypeError('"size" should be a scalar value.')
    elif not isinstance(size, (int, numpy.integer)):
        TypeError('"size" should be an integer.')
    elif size < 1:
        raise ValueError('"size" should be a positive integer.')

    # Check gram
    if gram is None:
        raise TypeError('"gram" cannot be None.')
    elif not numpy.isscalar(gram):
        raise TypeError('"gram" should be a scalar value.')
    elif not isinstance(gram, bool):
        raise TypeError('"gram" should be boolean.')

    # Check format
    if format is not None:
        if not isinstance(format, str):
            raise TypeError('"format" should be string.')
        elif format not in ['csr', 'csc']:
            raise ValueError('"format" should be either "csr" or "csc".')

    # Check dtype
    if dtype is None:
        raise TypeError('"dtype" cannot be None.')
    elif not numpy.isscalar(dtype):
        raise TypeError('"dtype" should be a scalar value.')
    elif not isinstance(dtype, str):
        raise TypeError('"dtype" should be a string')
    elif dtype not in [r'float32', r'float64', r'float128']:
        raise TypeError('"dtype" should be either "float32", "float64", or ' +
                        '"float128".')


# ==============
# toeplitz trace
# ==============

def toeplitz_trace(a, b, size, gram=False):
    """
    Compute the trace of the Toeplitz matrix using an analytic formula.

    The Toeplitz matrix using the entries :math:`a` and :math:`b` refers to the
    matrix generated by :func:`imate.toeplitz`.

    Parameters
    ----------

    a : float
        The diagonal elements of the Toeplitz matrix.

    b : float
        The upper off-diagonal elements of the Toeplitz matrix.

    size : int, default=20
        Size of the square matrix.

    gram : bool, default=False
        If `False`, the matrix is assumed to be bi-diagonal Toeplitz. If
        `True`, the Gramian of the matrix is considered instead.

    Returns
    -------

    trace : float
        Trace of Toeplitz matrix

    See Also
    --------

    imate.toeplitz
    imate.sample_matrices.toeplitz_logdet
    imate.sample_matrices.toeplitz_traceinv

    Notes
    -----

    For the matrix :math:`\\mathbf{A}` given in :func:`imate.toeplitz`, the
    trace is computed by

    .. math::

        \\mathrm{trace}(\\mathbf{A}) = n a,

    where :math:`n` is the size of the matrix. For the Gramian matrix,
    :math:`\\mathbf{B} = \\mathbf{A}^{\\intercal} \\mathbf{A}` (when ``gram``
    is set to `True`), the trace is

    .. math::

        \\mathrm{trace}(\\mathbf{B}) = a^2 + (n-1)(a^2 + b^2).

    Examples
    --------

    .. code-block:: python

        >>> from imate.sample_matrices import toeplitz_trace
        >>> a, b = 2, 3

        >>> toeplitz_trace(a, b, size=6)
        12

        >>> toeplitz_trace(a, b, size=6, gram=True)
        69
    """

    if gram:
        trace_ = a**2 + (size-1) * (a**2 + b**2)

    else:
        trace_ = size * a

    return trace_


# =================
# toeplitz traceinv
# =================

def toeplitz_traceinv(a, b, size, gram=False):
    """
    Computes the trace of the inverse of Toeplitz matrix using an analytic
    formula.

    The Toeplitz matrix using the entries :math:`a` and :math:`b` refers to the
    matrix generated by :func:`imate.toeplitz`.

    Parameters
    ----------

    a : float
        The diagonal elements of the Toeplitz matrix.

    b : float
        The upper off-diagonal elements of the Toeplitz matrix.

    size : int, default=20
        Size of the square matrix.

    gram : bool, default=False
        If `False`, the  matrix is assumed to be bi-diagonal Toeplitz. If
        `True`, the Gramian of the matrix is considered instead.

    Returns
    -------

    traceinv : float
        Trace of the inverse of Toeplitz matrix

    See Also
    --------

    imate.toeplitz
    imate.sample_matrices.toeplitz_logdet
    imate.sample_matrices.toeplitz_trace

    Notes
    -----

    For the matrix :math:`\\mathbf{A}` given in :func:`imate.toeplitz`, the
    trace of its inverse is computed by

    .. math::

        \\mathrm{trace}(\\mathbf{A}^{-1}) = \\frac{n}{a},

    where :math:`n` is the size of the matrix. For the Gramian matrix,
    :math:`\\mathbf{B} = \\mathbf{A}^{\\intercal} \\mathbf{A}` (when ``gram``
    is set to `True`), the trace of inverse is

    .. math::

        \\mathrm{trace}(\\mathbf{B}^{-1}) =
        \\begin{cases}
            \\displaystyle{\\frac{n(n+1)}{2 a^2}}, & \\text{if} a=b \\

            \\displaystyle{\\frac{1}{a^2 - b^2}
            \\frac{q^2 (q^{2n} - 1)}{q^2-1}}
            , &
            \\text{if} a \\neq b
        \\end{cases}

    where :math:`q = b/a`. If :math:`n \\gg 1`, then for :math:`q \\neq 1` we
    have

    .. math::
        :label: limit

        \\mathrm{trace}(\\mathbf{B}^{-1}) \\approx \\frac{1}{a^2 - b^2}
        \\left( n - \\frac{q^{2}}{1 - q^2} \\right).

    This function uses the approximation :math:numref:`limit` when
    :math:`n > 200`.

    Examples
    --------

    .. code-block:: python

        >>> from imate.sample_matrices import toeplitz_traceinv
        >>> a, b = 2, 3

        >>> toeplitz_traceinv(a, b, size=6)
        3.0

        >>> toeplitz_traceinv(a, b, size=6, gram=True)
        45.148681640625
    """

    if gram:
        if a == b:
            traceinv_ = size * (size+1) / (2.0 * a**2)
        else:
            q = b / a
            if size < 200:
                traceinv_ = (1.0 / (a**2 - b**2)) * \
                    (size - (q**2) * ((q**(2.0*size) - 1.0) / (q**2 - 1.0)))
            else:
                # Using asymptotic approximation of large sums
                traceinv_ = (1.0 / (a**2 - b**2)) * \
                            (size - ((q**2) / (1.0 - q**2)))
    else:
        traceinv_ = size / a

    return traceinv_


# ===============
# toeplitz logdet
# ===============

def toeplitz_logdet(a, b, size, gram=False):
    """
    Compute the log-determinant of Toeplitz matrix using an analytic formula.

    The Toeplitz matrix using the entries :math:`a` and :math:`b` refers to the
    matrix generated by :func:`imate.toeplitz`.

    Parameters
    ----------

    a : float
        The diagonal elements of the Toeplitz matrix.

    b : float
        The upper off-diagonal elements of the Toeplitz matrix.

    size : int, default=20
        Size of the square matrix.

    gram : bool, default=False
        If `False`, the matrix is assumed to be bi-diagonal Toeplitz. If
        `True`, the Gramian of the matrix is considered instead.

    Returns
    -------

    logdet : float
        Natural logarithm of the determinant of Toeplitz matrix

    See Also
    --------

    imate.toeplitz
    imate.sample_matrices.toeplitz_trace
    imate.sample_matrices.toeplitz_traceinv

    Notes
    -----

    For the matrix :math:`\\mathbf{A}` given in :func:`imate.toeplitz`, the
    log-determinant is

    .. math::

        \\mathrm{logdet}(\\mathbf{A}) = n \\log_e(a),

    where :math:`n` is the size of the matrix. For the Gramian matrix,
    :math:`\\mathbf{B} = \\mathbf{A}^{\\intercal} \\mathbf{A}` (when ``gram``
    is set to `True`), the log-determinant is

    .. math::

        \\mathrm{logdet}(\\mathbf{B}) = 2 n \\log_e(a)

    Examples
    --------

    .. code-block:: python

        >>> from imate.sample_matrices import toeplitz_logdet
        >>> a, b = 2, 3

        >>> toeplitz_logdet(a, b, size=6)
        4.1588830833596715

        >>> toeplitz_logdet(a, b, size=6, gram=True)
        8.317766166719343
    """

    logdet_ = size * numpy.log(a)
    if gram:
        logdet_ = 2.0 * logdet_

    return logdet_


# =================
# toeplitz schatten
# =================

def toeplitz_schatten(a, b, size, p=2):
    """
    Compute the Schatten norm of Toeplitz matrix using an analytic formula.

    The Toeplitz matrix using the entries :math:`a` and :math:`b` refers to the
    matrix generated by :func:`imate.toeplitz`.

    Parameters
    ----------

    a : float
        The diagonal elements of the Toeplitz matrix.

    b : float
        The upper off-diagonal elements of the Toeplitz matrix.

    size : int, default=20
        Size of the square matrix.

    p : float, default=2
        The order :math:`p` of Schatten :math:`p`-norm.

    Returns
    -------

    logdet : float
        Natural logarithm of the determinant of Toeplitz matrix

    See Also
    --------

    imate.toeplitz
    imate.sample_matrices.toeplitz_logdet
    imate.sample_matrices.toeplitz_trace
    imate.sample_matrices.toeplitz_traceinv

    Notes
    -----

    The Schatten :math:`p`-norm of matrix :math:`\\mathbf{A}` is defined as

    .. math::
        :label: schatten-eq

        \\Vert \\mathbf{A} \\Vert_p =
        \\begin{cases}
            \\left| \\mathrm{det}(\\mathbf{A})
            \\right|^{\\frac{1}{n}}, & p=0, \\\\
            \\left| \\frac{1}{n}
            \\mathrm{trace}(\\mathbf{A}^{p})
            \\right|^{\\frac{1}{p}}, & p \\neq 0,
        \\end{cases}

    where :math:`n` is the size of the matrix. When :math:`p \\geq 0`, the
    above definition is the Schatten **norm**, and when :math:`p < 0`, the
    above is the Schatten **anti-norm**.

    .. note::

        Conventionally, the Schatten norm is defined without the normalizing
        factor :math:`\\frac{1}{n}` in :math:numref:`schatten-eq`. However,
        this factor is justified by the continuity granted by

        .. math::

            \\lim_{p \\to 0} \\Vert \\mathbf{A} \\Vert_p =
            \\Vert \\mathbf{A} \\Vert_0.

    For the matrix :math:`\\mathbf{A}` given in :func:`imate.toeplitz`, the
    Schatten norm (or anti-norm) is

    .. math::

        \\Vert \\mathbf{A} \\Vert_p = a.

    Examples
    --------

    .. code-block:: python

        >>> from imate.sample_matrices import toeplitz_schatten
        >>> a, b = 2, 3

        >>> # Schatten 2-norm
        >>> toeplitz_schatten(a, b, size=6)
        2.0
    """

    return a
