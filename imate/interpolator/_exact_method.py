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

from __future__ import print_function
from ._interpolant_base import InterpolantBase


# ============
# Exact Method
# ============

class ExactMethod(InterpolantBase):
    """
    Evaluates Schatten norm (or anti-norm) of an affine matrix function (no
    interpolation).

    This class does not interpolate. Rather, it only returns the exact
    function value by calling :func:`imate.schatten`. The output of this
    function could be used as a benchmark to test the other interpolation
    methods.

    Parameters
    ----------

    A : numpy.ndarray, scipy.sparse matrix
        A square matrix. Matrix can be dense or sparse.

    B : numpy.ndarray, scipy.sparse matrix, default=None
        A square matrix of the same type and size as `A`.

    p : float, default=2
        The order :math:`p` in the Schatten :math:`p`-norm which can be real
        positive, negative or zero.

    options : dict, default={}
        The Schatten norm is computed using :func:`imate.schatten` function
        which itself calls either of

        * :func:`imate.logdet` (if :math:`p=0`)
        * :func:`imate.trace` (if :math:`p>0`)
        * :func:`imate.traceinv` (if :math:`p < 0`).

        The ``options`` passes a dictionary of arguments to the above
        functions.

    verbose : bool, default=False
        If `True`, it prints some information about the computation process.

    See Also
    --------

    imate.InterpolateTrace
    imate.InterpolateLogdet
    imate.schatten

    Attributes
    ----------

    kind : str
        Method of interpolation. For this class, ``kind`` is ``ext``.

    verbose : bool
        Verbosity of the computation process

    n : int
        Size of the matrix

    q : int
        Number of interpolant points. For this class, `q` is zero.

    p : float
        Order of Schatten :math:`p`-norm

    Methods
    -------

    __call__
        See :meth:`imate.InterpolateSchatten.__call__`.
    eval
        See :meth:`imate.InterpolateSchatten.eval`.
    interpolate
        See :meth:`imate.InterpolateSchatten.interpolate`.
    bound
        See :meth:`imate.InterpolateSchatten.bound`.
    upper_bound
        See :meth:`imate.InterpolateSchatten.upper_bound`.
    plot
        See :meth:`imate.InterpolateSchatten.plot`.

    Notes
    -----

    **Schatten Norm:**

    In this class, the Schatten :math:`p`-norm of the matrix
    :math:`\\mathbf{A}` is defined by

    .. math::
        :label: schatten-eq-9

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
        factor :math:`\\frac{1}{n}` in :math:numref:`schatten-eq-9`. However,
        this factor is justified by the continuity granted by

        .. math::
            :label: schatten-continuous-9

            \\lim_{p \\to 0} \\Vert \\mathbf{A} \\Vert_p =
            \\Vert \\mathbf{A} \\Vert_0.

        See [1]_ (Section 2) and the examples in :func:`imate.schatten` for
        details.

    **Affine Matrix Function:**

    This class evaluates the one-parameter matrix function:

    .. math::

        \\tau_p: t \\mapsto \\| \\mathbf{A} + t \\mathbf{B} \\|_p,

    where :math:`t` is a real parameter.

    References
    ----------

    .. [1] Ameli, S., and Shadden. S. C. (2022). Interpolating Log-Determinant
           and Trace of the Powers of Matrix
           :math:`\\mathbf{A} + t \\mathbf{B}`.
           *Statistics and Computing* 32, 108.
           `https://doi.org/10.1007/s11222-022-10173-4
           <https://doi.org/10.1007/s11222-022-10173-4>`_.

    Examples
    --------

    **Basic Usage:**

    Evaluate the Schatten `2`-norm of the affine matrix function
    :math:`\\mathbf{A} + t \\mathbf{I}`:

    .. code-block:: python
        :emphasize-lines: 7, 12

        >>> # Generate two sample matrices (symmetric and positive-definite)
        >>> from imate.sample_matrices import correlation_matrix
        >>> A = correlation_matrix(size=20, scale=1e-1)

        >>> # Initialize interpolator object
        >>> from imate import InterpolateSchatten
        >>> f = InterpolateSchatten(A, p=2, kind='ext')

        >>> # Evaluate at inquiry point t = 0.4
        >>> t = 4e-1
        >>> f(t)
        1.7175340160001527

    Alternatively, call :meth:`imate.InterpolateSchatten.eval` to
    evaluate at points `t`:

    .. code-block:: python

        >>> # This is the same as f(t)
        >>> f.eval(t)
        1.7175340160001527

    **Passing Options:**

    The above examples, the internal computation is passed to
    :func:`imate.trace` function since :math:`p=2` is positive. You can pass
    arguments to the latter function using ``options`` argument. To do so,
    create a dictionary with the keys as the name of the argument. For
    instance, to use :ref:`imate.trace.slq` method with ``min_num_samples=20``
    and ``max_num_samples=100``, create the following dictionary:

    .. code-block:: python

        >>> # Specify arguments as a dictionary
        >>> options = {
        ...     'method': 'slq',
        ...     'min_num_samples': 20,
        ...     'max_num_samples': 100
        ... }

        >>> # Pass the options to the interpolator
        >>> f = InterpolateSchatten(A, B, options=options, kind='ext')
        >>> f(t)
        1.7047510802581667

    **Evaluate on Range of Points:**

    Evaluate an array of inquiry points ``t_array``:

    .. code-block:: python

        >>> # Initialize interpolator object
        >>> from imate import InterpolateSchatten
        >>> f = InterpolateSchatten(A, kind='ext')

        >>> # Interpolate at an array of points
        >>> import numpy
        >>> t_array = numpy.logspace(-2, 1, 1000)
        >>> norm_array = f(t_array)

    **Plotting:**

    To plot the function, call :meth:`imate.InterpolateSchatten.plot` method.
    To compare with the true function values, pass ``compare=True`` argument.

    .. code-block:: python

        >>> f.plot(t_array, compare=True)

    .. image:: ../_static/images/plots/interpolate_schatten_ext_eig.png
        :align: center
        :class: custom-dark

    Since the `ext` method exactly evaluates the function (without
    interpolation), the error of the result is zero, as shown on the
    right-hand side plot.
    """

    # ====
    # Init
    # ====

    def __init__(self, A, B=None, p=2, options={}, verbose=False):
        """
        Initializes the parent class.
        """

        # Base class constructor
        super(ExactMethod, self).__init__(
                A, B=B, p=p, ti=None, options=options, verbose=verbose)

        # Attributes
        self.q = 0

    # ===========
    # Interpolate
    # ===========

    def interpolate(self, t):
        """
        This function does not interpolate, rather evaluates the exact function
        value.

        For this method (``kind=ext``), this function calls
        :meth:`imate.InterpolateSchatten.eval` method.

        Parameters
        ----------

        t : float or array_like[float]
            An inquiry point (or list of points) to interpolate.

        Returns
        -------

        norm : float or numpy.array
            Exact values of the function. If the input `t` is a list or array,
            the output is an array of the same size of `t`.

        See Also
        --------

        imate.InterpolateSchatten.__call__
        imate.InterpolateSchatten.eval

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12, 17

            >>> # Generate two sample matrices (symmetric positive-definite)
            >>> from imate.sample_matrices import correlation_matrix
            >>> A = correlation_matrix(size=20, scale=1e-1)

            >>> # Initialize interpolator object
            >>> from imate import InterpolateSchatten
            >>> f = InterpolateSchatten(A, kind='ext')

            >>> # Exact function value at an inquiry point t = 0.4
            >>> t = 4e-1
            >>> f.interpolate(t)
            1.7175340160001527

            >>> # The above is the same as eval
            >>> f.eval(t)
            1.7175340160001527
        """

        # Do not interpolate, instead compute the exact value
        return self.eval(t)
