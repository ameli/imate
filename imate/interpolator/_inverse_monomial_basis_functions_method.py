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

import numpy


# =======================================
# Inverse Monomial Basis Functions Method
# =======================================

class InverseMonomialBasisFunctionsMethod(InterpolantBase):
    """
    Interpolate Schatten norm (or anti-norm) of an affine matrix function using
    inverse monomial basis functions (IMBF) method.

    Parameters
    ----------

    A : numpy.ndarray, scipy.sparse matrix
        Symmetric positive-definite matrix (positive-definite if `p` is
        non-positive). Matrix can be dense or sparse.

        .. warning::

            Symmetry and positive (semi-) definiteness of `A` will not be
            checked. Make sure `A` satisfies these conditions.

    B : numpy.ndarray, scipy.sparse matrix, default=None
        Symmetric positive-definite matrix (positive-definite if `p` is
        non-positive). Matrix can be dense or sparse. `B` should have the same
        size and type of `A`. If `B` is `None` (default value), it is assumed
        that `B` is the identity matrix.

        .. warning::

            Symmetry and positive (semi-) definiteness of `B` will not be
            checked. Make sure `B` satisfies these conditions.

    p : float, default=2
        The order :math:`p` in the Schatten :math:`p`-norm which can be real
        positive, negative or zero.

    options : dict, default={}
        At each interpolation point :math:`t_i`, the Schatten norm is computed
        using :func:`imate.schatten` function which itself calls either of

        * :func:`imate.logdet` (if :math:`p=0`)
        * :func:`imate.trace` (if :math:`p>0`)
        * :func:`imate.traceinv` (if :math:`p < 0`).

        The ``options`` passes a dictionary of arguments to the above
        functions.

    verbose : bool, default=False
        If `True`, it prints some information about the computation process.

    ti : float or array_like(float), default=None
        Interpolation points, which can be a single number, a list or an array
        of interpolation points. The interpolator honors the exact function
        values at the interpolant points.

    basis_func_type: {'non-ortho', 'ortho', 'ortho2'}, default='ortho2'
        Type of basis functions (see Notes below).

    See Also
    --------

    imate.InterpolateTrace
    imate.InterpolateLogdet
    imate.schatten

    Attributes
    ----------

    kind : str
        Method of interpolation. For this class, ``kind`` is ``imbf``.

    verbose : bool
        Verbosity of the computation process

    n : int
        Size of the matrix

    q : int
        Number of interpolant points.

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
        :label: schatten-eq-6

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
        factor :math:`\\frac{1}{n}` in :math:numref:`schatten-eq-6`. However,
        this factor is justified by the continuity granted by

        .. math::
            :label: schatten-continuous-6

            \\lim_{p \\to 0} \\Vert \\mathbf{A} \\Vert_p =
            \\Vert \\mathbf{A} \\Vert_0.

        See [1]_ (Section 2) and the examples in :func:`imate.schatten` for
        details.

    **Interpolation of Affine Matrix Function:**

    This class interpolates the one-parameter matrix function:

    .. math::

        t \\mapsto \\| \\mathbf{A} + t \\mathbf{B} \\|_p,

    where the matrices :math:`\\mathbf{A}` and :math:`\\mathbf{B}` are
    symmetric and positive semi-definite (positive-definite if :math:`p < 0`)
    and :math:`t \\in [t_{\\inf}, \\infty)` is a real parameter where
    :math:`t_{\\inf}` is the minimum :math:`t` such that
    :math:`\\mathbf{A} + t_{\\inf} \\mathbf{B}` remains positive-definite.

    **Method of Interpolation:**

    Define the function

    .. math::

        \\tau_p(t) = \\frac{\\Vert \\mathbf{A} + t \\mathbf{B} \\Vert_p}
        {\\Vert \\mathbf{B} \\Vert_p},

    and :math:`\\tau_{p, 0} = \\tau_p(0)`. Then, we approximate
    :math:`\\tau_p(t)` by

    .. math::

        \\tau_p(t) \\approx \\tau_{p, 0} + \\sum_{i = 0}^q w_i
        \\phi_i(t),

    where :math:`\\phi_i` are some known basis functions, and :math:`w_i` are
    the coefficients of the linear basis functions. The first coefficient is
    set to :math:`w_{0} = 1` and the rest of the weights are to be found from
    the known function values :math:`\\tau_{p, i} = \\tau_p(t_i)` at the given
    interpolant points :math:`t_i`.

    **Interpolation Points:**

    The best practice is to provide an array of interpolation points that are
    equally distanced on the logarithmic scale. For instance, to produce four
    interpolation points in the interval :math:`[10^{-2}, 1]`:

    .. code-block:: python

        >>> import numpy
        >>> ti = numpy.logspace(-2, 1, 4)

    **Basis Functions:**

    In this module, three kinds of basis functions which can be set by the
    argument ``basis_func_type``.

    When ``basis_func_type`` is set to ``non-ortho``, the basis
    functions are the inverse of the monomial functions defined by

    .. math::

        \\phi_i(t) = t^{\\frac{1}{i+1}}, \\qquad i = 0, \\dots, q.

    .. warning::

        The non-orthogonal basis functions can lead to ill-conditioned system
        of equations for finding the weight coefficients :math:`w_i`. When the
        number of interpolating points is large (such as :math:`q > 6`), it is
        recommended to use the orthogonalized set of basis functions described
        next.

    When ``basis_func_type`` is set to ``'ortho'`` or
    ``'ortho2'``, the orthogonal form of the above basis functions are
    used. Orthogonal basis functions are formed by the above non-orthogonal
    functions as

    .. math::

        \\phi_i^{\\perp}(t) = \\alpha_i \\sum_{j=1}^i a_{ij} \\phi_j(t)

    The coefficients :math:`\\alpha_i` and :math:`a_{ij}` can be obtained by
    the python package `ortho
    <https://ameli.github.io/ortho>`_. These coefficients are
    hard-coded in this function up to :math:`i = 9`. Thus, in this module, up
    to nine interpolant points are supported.

    The difference between ``ortho`` and ``orth2`` basis functions is that
    in the former, the functions :math:`\\phi_i` for all :math:`i=0,\\dots, q`
    are orthogonalized, whereas in the latter, only the functions
    :math:`i=1,\\dots,q` (excluding :math:`i=0`) are orthogonalized.

    .. note::

        The recommended basis function type is ``'ortho2'``.

    **How to Generate the Basis Functions:**

    The coefficients :math:`\\alpha` and :math:`a` of the basis functions are
    hard-coded up to :math:`q=9` in this class. To
    generate further basis functions, use `ortho python package
    <https://ameli.github.io/ortho>`_.

    Install this package by

    ::

        pip install ortho

    * To generate the coefficients corresponding to ``ortho`` basis:

      ::

          gen-orth -n 9 -s 0

    * To generate the coefficients corresponding to ``ortho2`` basis:

      ::

          gen-orth -n 9 -s 1

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

    Interpolate the Schatten `2`-norm of the affine matrix function
    :math:`\\mathbf{A} + t \\mathbf{B}` using ``imbf`` algorithm and the
    interpolating points :math:`t_i = [10^{-2}, 10^{-1}, 1, 10]`.

    .. code-block:: python
        :emphasize-lines: 9, 13

        >>> # Generate two sample matrices (symmetric and positive-definite)
        >>> from imate.sample_matrices import correlation_matrix
        >>> A = correlation_matrix(size=20, scale=1e-1)
        >>> B = correlation_matrix(size=20, scale=2e-2)

        >>> # Initialize interpolator object
        >>> from imate import InterpolateSchatten
        >>> ti = [1e-2, 1e-1, 1, 1e1]
        >>> f = InterpolateSchatten(A, B, p=2, kind='imbf', ti=ti)

        >>> # Interpolate at an inquiry point t = 0.4
        >>> t = 4e-1
        >>> f(t)
        1.7328745175033962

    Alternatively, call :meth:`imate.InterpolateSchatten.interpolate` to
    interpolate at points `t`:

    .. code-block:: python

        >>> # This is the same as f(t)
        >>> f.interpolate(t)
        1.7328745175033962

    To evaluate the exact value of the Schatten norm at point `t` without
    interpolation, call :meth:`imate.InterpolateSchatten.eval` function:

    .. code-block:: python

        >>> # This evaluates the function value at t exactly (no interpolation)
        >>> f.eval(t)
        1.7374809371539666

    It can be seen that the relative error of interpolation compared to the
    exact solution in the above is :math:`0.26 \\%` using only four
    interpolation points :math:`t_i`, which is a remarkable result.

    .. warning::

        Calling :meth:`imate.InterpolateSchatten.eval` may take a longer time
        to compute as it computes the function exactly. Particularly, if `t` is
        a large array, it may take a very long time to return the exact values.

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
        >>> f = InterpolateSchatten(A, B, p=2, options=options, kind='imbf',
        ...                         ti=ti)
        >>> f(t)
        1.7397564159794918

    You may get a different result than the above as the `slq` method is a
    randomized method.

    **Interpolate on Range of Points:**

    Once the interpolation object ``f`` in the above example is
    instantiated, calling :meth:`imate.InterpolateSchatten.interpolate` on
    a list of inquiry points `t` has almost no computational cost. The next
    example inquires interpolation on `1000` points `t`:

    Interpolate an array of inquiry points ``t_array``:

    .. code-block:: python

        >>> # Create an interpolator object again
        >>> ti = 1e-1
        >>> f = InterpolateSchatten(A, B, kind='imbf', ti=ti)

        >>> # Interpolate at an array of points
        >>> import numpy
        >>> t_array = numpy.logspace(-2, 1, 1000)
        >>> norm_array = f.interpolate(t_array)

    **Plotting Interpolation and Compare with Exact Solution:**

    To plot the interpolation results, call
    :meth:`imate.InterpolateSchatten.plot` function. To compare with the true
    values (without interpolation), pass ``compare=True`` to the above
    function.

    .. warning::

        By setting ``compare`` to `True`, every point in the array `t` is
        evaluated both using interpolation and with the exact method (no
        interpolation). If the size of `t` is large, this may take a very
        long run time.

    .. code-block:: python

        >>> f.plot(t_array, normalize=True, compare=True)

    .. image:: ../_static/images/plots/interpolate_schatten_imbf.png
        :align: center
        :class: custom-dark

    From the error plot in the above, it can be seen that with only four
    interpolation points, the error of interpolation for a wide range of
    :math:`t` is no more than :math:`0.3 \\%`. Also, note that the error on the
    interpolant points :math:`t_i=[10^{-2}, 10^{-1}, 1, 10]` is zero since the
    interpolation scheme honors the exact function value at the interpolation
    points.
    """

    # ====
    # Init
    # ====

    def __init__(self, A, B=None, p=2, options={}, verbose=False, ti=[],
                 basis_func_type='ortho2'):
        """
        Initializes the base class and the attributes, namely, the computes the
        trace at interpolant points.
        """

        if (ti is None) or (ti == []):
            raise ValueError('"ti" should be a list or array.')

        # Base class constructor
        super(InverseMonomialBasisFunctionsMethod, self).__init__(
                A, B=B, p=p, ti=ti, options=options, verbose=verbose)

        self.basis_func_type = basis_func_type

        # Initialize Interpolator
        self.alpha = None
        self.a = None
        self.w = None
        self.initialize_interpolator()

    # =======================
    # initialize interpolator
    # =======================

    def initialize_interpolator(self):
        """
        Internal function that is called by the class constructor. It computes
        the weight coefficients :math:`w_i` and stores them in the member
        variable ``self.w``.
        """

        if self.verbose:
            print('Initialize interpolator ...')

        # Method 1: Use non-orthogonal basis functions
        if self.basis_func_type == 'non-ortho':
            # Form a linear system for weights w
            b = self.tau_i - self.tau0 - self.t_i
            C = numpy.zeros((self.q, self.q))
            for i in range(self.q):
                for j in range(self.q):
                    C[i, j] = self.basis_functions(j, self.t_i[i])

            if self.verbose:
                print('Condition number: %f' % (numpy.linalg.cond(C)))

            self.w = numpy.linalg.solve(C, b)

        elif self.basis_func_type == 'ortho':

            # Method 2: Use orthogonal basis functions
            self.alpha, self.a = self.orthogonal_basis_function_coefficients()

            if self.alpha.size < self.t_i.size:
                raise ValueError('Cannot regress order higher than %d. ' +
                                 'Decrease the number of interpolation ' +
                                 'points.' % (self.alpha.size))

            # Form a linear system Cw = b for weights w
            b = numpy.zeros(self.q+1)
            b[:-1] = self.tau_i - self.tau0
            b[-1] = 1.0
            C = numpy.zeros((self.q+1, self.q+1))
            for i in range(self.q):
                for j in range(self.q+1):
                    C[i, j] = self.basis_functions(j, self.t_i[i]/self.scale_t)

            # The coefficient of term "t" should be 1.
            C[-1, :] = self.alpha[:self.q+1]*self.a[:self.q+1, 0]

            if self.verbose:
                print('Condition number: %f' % (numpy.linalg.cond(C)))

            # Solve weights
            self.w = numpy.linalg.solve(C, b)

        elif self.basis_func_type == 'ortho2':

            # Method 3: Use orthogonal basis functions
            self.alpha, self.a = self.orthogonal_basis_function_coefficients()

            if self.alpha.size < self.t_i.size:
                raise ValueError('Cannot regress order higher than %d. ' +
                                 'Decrease the number of interpolation ' +
                                 'points.' % (self.alpha.size))

            # Form a linear system Aw = b for weights w
            b = self.tau_i - self.tau0 - self.t_i
            C = numpy.zeros((self.q, self.q))
            for i in range(self.q):
                for j in range(self.q):
                    C[i, j] = self.basis_functions(j, self.t_i[i]/self.scale_t)

            if self.verbose:
                print('Condition number: %f' % (numpy.linalg.cond(C)))

            # Solve weights
            self.w = numpy.linalg.solve(C, b)
            # Lambda = 1e1   # Regularization parameter  # SETTING
            # C2 = C.T.dot(C) + Lambda * numpy.eye(C.shape[0])
            # b2 = C.T.dot(b)
            # self.w = numpy.linalg.solve(C2, b2)

        if self.verbose:
            print('Done.')

    # ===
    # Phi
    # ===

    @staticmethod
    def phi(i, t):
        """
        Non-orthogonal basis function, which is defined by

        .. math::

            \\phi_i(t) = t^{\\frac{1}{i}}, \\qquad i > 0.

        :param t: Inquiry point.
        :type t: float

        :return: The value of the function :math:`\\phi(t)`
        :rtype: float
        """

        return t**(1.0/i)

    # ===============
    # Basis Functions
    # ===============

    def basis_functions(self, j, t):
        """
        Returns the basis functions at inquiry point :math:`t`

        The index j of the basis functions should start from 1.

        :param t: Inquiry point.
        :type t: float

        :return: Basis functions at inquiry point.
        :rtype: float

        Depending on ``basis_func_type``, the basis functions are:

        * For ``NonOrthogonal``:

            .. math::

                \\phi_i(t) = t^{\\frac{1}{i}}, \\qquad i > 0

        * For ``Orthogonal``:

            .. math::

                \\phi_i^{\\perp}(t) = \\alpha_i \\sum_{j=1}^9 a_{ij} \\phi_j(t)

        * For ``Orthogona2``:

            .. math::

                \\phi_i^{\\perp}(t) = \\alpha_i \\sum_{j=1}^9 a_{ij}
                \\phi_{j+1}(t)

        .. note::

            The difference between ``Orthogonal`` and ``Orthogonal2`` is that
            in the former, the functions :math:`\\phi_j^{\\perp}` at
            :math:`j=1, \\dots, 9` are orthogonal but in the latter, the
            functions at :math:`j=2, \\dots, 9` are orthogonal. That is they
            are not orthogonal to :math:`\\phi_1(t) = t`.
        """

        if self.basis_func_type == 'non-ortho':
            return InverseMonomialBasisFunctionsMethod.phi(j+2, t)

        elif self.basis_func_type == 'ortho':

            # Use Orthogonal basis functions
            alpha, a = self.orthogonal_basis_function_coefficients()

            phi_perp = 0
            for i in range(a.shape[1]):
                phi_perp += alpha[j]*a[j, i] * \
                        InverseMonomialBasisFunctionsMethod.phi(i+1, t)

            return phi_perp

        elif self.basis_func_type == 'ortho2':

            # Use Orthogonal basis functions
            alpha, a = self.orthogonal_basis_function_coefficients()

            phi_perp = 0
            for i in range(a.shape[1]):
                phi_perp += alpha[j]*a[j, i] * \
                    InverseMonomialBasisFunctionsMethod.phi(i+2, t)

            return phi_perp

        else:
            raise ValueError('Method is invalid.')

    # ======================================
    # orthogonal basis function coefficients
    # ======================================

    def orthogonal_basis_function_coefficients(self):
        """
        Hard-coded coefficients :math:`\\alpha_i` and :math:`a_{ij}` which will
        be used by :func:`basis_functions` to form the orthogonal basis:

        .. math::

            \\phi_i^{\\perp}(t) = \\alpha_i \\sum_{j=0}^9 a_{ij} \\phi_j(t).

        **Generate coefficients:**

        To generate these coefficients, see the python package
        `Orthogonal Functions <https://ameli.github.io/ortho>`_.

        Install this package by

            ::

                pip install ortho

        * To generate the coefficients corresponding to ``ortho`` basis:

          ::

            gen-orth -n 9 -s 0

        * To generate the coefficients corresponding to ``ortho2`` basis:

          ::

            gen-orth -n 9 -s 1

        :return: Weight coefficients of the orthogonal basis functions.
        :rtype: numpy.array, numpy.ndarray
        """

        q = 9
        a = numpy.zeros((q, q), dtype=float)

        if self.basis_func_type == 'ortho':
            alpha = numpy.array([
                +numpy.sqrt(2.0/1.0),
                -numpy.sqrt(2.0/2.0),
                +numpy.sqrt(2.0/3.0),
                -numpy.sqrt(2.0/4.0),
                +numpy.sqrt(2.0/5.0),
                -numpy.sqrt(2.0/6.0),
                +numpy.sqrt(2.0/7.0),
                -numpy.sqrt(2.0/8.0),
                +numpy.sqrt(2.0/9.0)])

            a[0, :1] = numpy.array([1])
            a[1, :2] = numpy.array([4, -3])
            a[2, :3] = numpy.array([9, -18, 10])
            a[3, :4] = numpy.array([16, -60, 80, -35])
            a[4, :5] = numpy.array([25, -150, 350, -350, 126])
            a[5, :6] = numpy.array([36, -315, 1120, -1890, 1512, -462])
            a[6, :7] = numpy.array([49, -588, 2940, -7350, 9702, -6468, 1716])
            a[7, :8] = numpy.array([64, -1008, 6720, -23100, 44352, -48048,
                                   27456, -6435])
            a[8, :9] = numpy.array([81, -1620, 13860, -62370, 162162, -252252,
                                   231660, -115830, 24310])

        elif self.basis_func_type == 'ortho2':
            alpha = numpy.array([
                +numpy.sqrt(2.0/2.0),
                -numpy.sqrt(2.0/3.0),
                +numpy.sqrt(2.0/4.0),
                -numpy.sqrt(2.0/5.0),
                +numpy.sqrt(2.0/6.0),
                -numpy.sqrt(2.0/7.0),
                +numpy.sqrt(2.0/8.0),
                -numpy.sqrt(2.0/9.0),
                +numpy.sqrt(2.0/10.0)])

            a[0, :1] = numpy.array([1])
            a[1, :2] = numpy.array([6, -5])
            a[2, :3] = numpy.array([20, -40, 21])
            a[3, :4] = numpy.array([50, -175, 210, -84])
            a[4, :5] = numpy.array([105, -560, 1134, -1008, 330])
            a[5, :6] = numpy.array([196, -1470, 4410, -6468, 4620, -1287])
            a[6, :7] = numpy.array([336, -3360, 13860, -29568, 34320, -20592,
                                   5005])
            a[7, :8] = numpy.array([540, -6930, 37422, -108108, 180180,
                                   -173745, 90090, -19448])
            a[8, :9] = numpy.array([825, -13200, 90090, -336336, 750750,
                                   -1029600, 850850, -388960, 75582])

        return alpha, a

    # ===========
    # interpolate
    # ===========

    def interpolate(self, t):
        """
        Interpolates :math:`\\mathrm{trace} \\left( (\\mathbf{A} +
        t \\mathbf{B})^{-1} \\right)` at :math:`t`.

        This is the main interface function of this module and it is used after
        the interpolation
        object is initialized.

        :param t: The inquiry point(s).
        :type t: float, list, or numpy.array

        :return: The interpolated value of the trace.
        :rtype: float or numpy.array

        **Details:**

        Depending on the ``basis_func_type``, the interpolation is as
        follows:

        For ``'NonOrthogonal'`` basis:

        .. math::

            \\frac{1}{\\tau(t)} = \\frac{1}{\\tau_0} + t + \\sum_{j=1}^q w_j
            \\phi_j(t).

        For ``'Orthogonal'`` and ``'Orthogonal2'`` bases:

        .. math::

            \\frac{1}{\\tau(t)} = \\frac{1}{\\tau_0} + t + \\sum_{j=1}^q w_j
            \\phi_j(t).
        """

        # Interpolation
        if self.basis_func_type == 'non-ortho':

            S = 0.0
            for j in range(self.q):
                S += self.w[j] * self.basis_functions(j, t)
            tau = self.tau0 + S + t

        elif self.basis_func_type == 'ortho':

            S = 0.0
            for j in range(self.w.size):
                S += self.w[j] * self.basis_functions(j, t/self.scale_t)
            tau = self.tau0 + S

        elif self.basis_func_type == 'ortho2':

            S = 0.0
            for j in range(self.q):
                S += self.w[j] * self.basis_functions(j, t/self.scale_t)
            tau = self.tau0 + S + t

        schatten = tau * self.schatten_B
        return schatten
