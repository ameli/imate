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
import numpy
import scipy
from ._interpolant_base import InterpolantBase


# ==================
# Eigenvalues Method
# ==================

class EigenvaluesMethod(InterpolantBase):
    """
    Evaluates Schatten norm (or anti-norm) of an affine matrix function (no
    interpolation).

    .. note::

        This class does not interpolate. Rather, it only returns the exact
        function value, which could be used as a benchmark to test the other
        interpolation methods.

    Parameters
    ----------

    A : numpy.ndarray, scipy.sparse matrix
        A square matrix. Matrix can be dense or sparse.

    B : None
        In this method, `B` should always be `None`, indicating the matrix `B`
        is the identity matrix.

    p : float, default=2
        The order :math:`p` in the Schatten :math:`p`-norm which can be real
        positive, negative or zero.

    options : dict, default={}
        At each interpolation point :math:`t_i`, the value of the Schatten norm
        is computed using :func:`imate.schatten` function which itself calls
        either of

        * :func:`imate.logdet` (if :math:`p=0`)
        * :func:`imate.trace` (if :math:`p>0`)
        * :func:`imate.traceinv` (if :math:`p < 0`).

        To pass extra parameters to the above functions, pass a dictionary of
        function arguments to ``options``.

    verbose : bool, default=False
        If `True`, it prints some information about the computation process.

    non_zero_ratio : float, default=0.9
        The ratio of the number of eigenvalues to be assumed non-zero over all
        eigenvalues. This option is only used for sparse matrices where not all
        its eigenvalues can be computed, rather, it is assumed some of
        the eigenvalues are very small and ignored.

    tol : float, default=1e-3
        Tolerance of computing eigenvalues. This option is only used for sparse
        matrices.

    Raises
    ------

    ValueError
        If `B` is not `None`.

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
        :label: schatten-eq-4

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
        factor :math:`\\frac{1}{n}` in :math:numref:`schatten-eq-4`. However,
        this factor is justified by the continuity granted by

        .. math::
            :label: schatten-continuous-4

            \\lim_{p \\to 0} \\Vert \\mathbf{A} \\Vert_p =
            \\Vert \\mathbf{A} \\Vert_0.

        See [1]_ (Section 2) and the examples in :func:`imate.schatten` for
        details.

    **Affine Matrix Function:**

    This class evaluates the one-parameter matrix function:

    .. math::

        \\tau_p: t \\mapsto \\| \\mathbf{A} + t \\mathbf{I} \\|_p,

    where :math:`t` is a real parameter and :math:`\\mathbf{I}` is the
    identity matrix.

    **Method of Evaluation:**

    This class uses the eigenvalues, :math:`\\lambda_i`, of the matrix
    :math:`\\mathbf{A}` to compute :math:`\\tau_p(t)` as follows:

    .. math::

        \\tau_p(t) =
        \\begin{cases}
        \\left( \\prod_{i=1}^n (\\lambda_i + t) \\right)^{\\frac{1}{n}},
        & p=0, \\\\
        \\left( \\frac{1}{n} \\sum_{i=1}^n (\\lambda_i + t)^p
        \\right)^{\\frac{1}{p}}, & p \\neq 0.
        \\end{cases}

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
        >>> f = InterpolateSchatten(A, p=2, kind='eig')

        >>> # Evaluate at inquiry point t = 0.4
        >>> t = 4e-1
        >>> f(t)
        1.7175340160001518

    Alternatively, call :meth:`imate.InterpolateSchatten.eval` to
    evaluate at points `t`:

    .. code-block:: python

        >>> # This is the same as f(t)
        >>> f.eval(t)
        1.7175340160001518

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
        >>> f = InterpolateSchatten(A, B, options=options, kind='eig')
        >>> f(t)
        1.7175340160001518

    **Evaluate on Range of Points:**

    Evaluate an array of inquiry points ``t_array``:

    .. code-block:: python

        >>> # Initialize interpolator object
        >>> from imate import InterpolateSchatten
        >>> f = InterpolateSchatten(A, B, kind='eig')

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

    Since the `eig` method exactly evaluates the function (without
    interpolation), the error of the result is zero, as shown on the
    right-hand side plot.
    """

    # ====
    # Init
    # ====

    def __init__(self, A, B=None, p=2, options={}, verbose=False,
                 non_zero_ratio=0.9, tol=1e-3):
        """
        Constructor of the class, which initializes the bases class and
        computes eigenvalues of the input matrices.
        """

        if B is not None:
            raise ValueError('In "EIG" method, "B" should be "None".')

        # Base class constructor
        super(EigenvaluesMethod, self).__init__(
                A, B=B, p=p, ti=None, options=options, verbose=verbose)

        # Attributes
        self.non_zero_ratio = non_zero_ratio
        self.tol = tol
        self.q = 0

        # Initialize Interpolator
        self.A_eigenvalues = None
        self.B_eigenvalues = None
        self.initialize_interpolator()

    # =======================
    # initialize interpolator
    # =======================

    def initialize_interpolator(self):
        """
        Initializes the ``A_eigenvalues`` and ``B_eigenvalues``  member data of
        the class.

        .. note::

            If the matrix ``A`` is sparse, it is not possible to find all of
            its eigenvalues. We only find a fraction of the number of its
            eigenvalues with the larges magnitude and we assume the rest of the
            eigenvalues are close to zero.
        """

        if self.verbose:
            print('Initialize interpolator ...', end='')

        # Find eigenvalues of A
        if self.use_sparse:

            # A is sparse
            self.A_eigenvalues = numpy.zeros(self.n)

            # find 90% of eigenvalues, assume the rest are close to zero.
            NumNoneZeroEig = int(self.n*self.non_zero_ratio)
            self.A_eigenvalues[:NumNoneZeroEig] = scipy.sparse.linalg.eigsh(
                    self.A, NumNoneZeroEig, which='LM', tol=self.tol,
                    return_eigenvectors=False)

        else:
            # A is dense matrix
            self.A_eigenvalues = scipy.linalg.eigh(self.A, eigvals_only=True,
                                                   check_finite=False)

        # Find eigenvalues of B
        if self.B_is_identity:

            # B is identity
            self.B_eigenvalues = numpy.ones(self.n, dtype=float)

        else:
            # B is not identity
            if self.use_sparse:

                # B is sparse
                self.B_eigenvalues = numpy.zeros(self.n)

                # find 90% of eigenvalues, assume the rest are close to zero.
                NumNoneZeroEig = int(self.n*self.non_zero_ratio)
                self.B_eigenvalues[:NumNoneZeroEig] = \
                    scipy.sparse.linalg.eigsh(
                        self.B, NumNoneZeroEig, which='LM', tol=self.tol,
                        return_eigenvectors=False)

            else:
                # B is dense matrix
                self.B_eigenvalues = scipy.linalg.eigh(
                        self.B, eigvals_only=True, check_finite=False)

        if self.verbose:
            print(' Done.')

    # ===========
    # Interpolate
    # ===========

    def interpolate(self, t):
        """
        Computes the function :math:`\\mathrm{trace}\\left( (\\mathbf{A} +
        t \\mathbf{B})^{-1} \\right)` at the input point :math:`t` by

        .. math::

            \\mathrm{trace}\\left( (\\mathbf{A} + t \\mathbf{B})^{-1}
            \\right) = \\sum_{i = 1}^n \\frac{1}{\\lambda_i + t \\mu_i}

        where :math:`\\lambda_i` is the eigenvalue of :math:`\\mathbf{A}`.
        and  :math:`\\mu_i` is the eigenvalue of :math:`\\mathbf{B}`.

        :param: t: An inquiry point, which can be a single number, or an array
            of numbers.
        :type t: float or numpy.array

        :return: The exact value of the trace of inverse of ``A + tB``.
        :rtype: float
        """

        # Compute trace using eigenvalues
        eig_An = self.A_eigenvalues + t * self.B_eigenvalues

        if self.p == 0:
            # Compute logdet
            logdet_ = numpy.sum(numpy.log(numpy.abs(eig_An)))
            schatten = numpy.exp(logdet_/self.n)
        else:
            tracep = numpy.sum(eig_An**self.p)
            schatten = (tracep/self.n)**(1.0/self.p)

        return schatten
