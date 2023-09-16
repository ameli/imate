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

import numbers
import numpy
import scipy
import scipy.interpolate
import scipy.optimize
from scipy.special import eval_chebyt, eval_chebyu
from scipy.integrate import quad


# ===================================
# Chebyshev Rational Functions Method
# ===================================

class ChebyshevRationalFunctionsMethod(InterpolantBase):
    """
    Interpolate Schatten norm (or anti-norm) of an affine matrix function using
    Chebyshev rational functions (CRF) method.

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
        Interpolation points, which can be a list or an array of interpolation
        points. If an integer number is given, then `ti` number of
        interpolation points are automatically generated on Chebyshev points.
        The interpolator honors the exact function values at the interpolant
        points.

    scale : float, default=None
        A scalar value :math:`\\alpha` that scales the inputs :math:`t` to
        :math:`t / \\alpha`. If set to `None` it is estimated to optimally
        minimize the arc-length curvature of the interpolation curve.

    func_type: {1, 2}, default=1
        Type of interpolation function model. See Notes below for details.

    See Also
    --------

    imate.InterpolateTrace
    imate.InterpolateLogdet
    imate.schatten

    Attributes
    ----------

    kind : str
        Method of interpolation. For this class, ``kind`` is ``crf``.

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
    get_scale
        See :meth:`imate.InterpolateSchatten.get_scale`.
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
        :label: schatten-eq-7

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
        factor :math:`\\frac{1}{n}` in :math:numref:`schatten-eq-7`. However,
        this factor is justified by the continuity granted by

        .. math::
            :label: schatten-continuous-7

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
    :math:`\\tau_p(t)` as follows. Transform the data :math:`(t, \\tau_p)` to
    :math:`(x, y)` where

    .. math::

        x = \\frac{t - \\alpha}{t + \\alpha},

    where :math:`\\alpha` is given by the argument ``scale``. Also, if
    ``func_type=1``, then :math:`y` is defined by

    .. math::

        y = \\frac{\\tau_p(t)}{\\tau_{p, 0} + t} - 1,

    and, if ``func_type=2``, then :math:`y` is defined by

    .. math::

        y = \\frac{\\tau_p(t) -\\tau_{p, 0}}{t} - 1.

    The Chebyshev rational function method, interpolates the data
    :math:`(x, y_{\\alpha})` by

    .. math::

        y_{\\alpha} = \\sum_{i=1}^{q+1} \\frac{w_i(\\alpha)}{2}
        \\left(1 - T_i(x) \\right),

    where :math:`T_i` is the i-th Chebyshev polynomial, and :math:`q` is the
    interpolation order. The weight parameters :math:`w_i` for
    :math:`i=1,\\dots, q+1` are found by fitting the above function on the
    :math:`q+1` interpolation points.

    This method is called the Chebyshev rational function since the function

    .. math::

        T_i(x) = r_i\\left(\\frac{t - \\alpha}{t + \\alpha} \\right),

    is known as the Chebyshev rational function.

    **Scale Parameter:**

    If the scale parameter :math:`\\alpha` is not given (that is,
    ``scale=None``), then this parameter is automatically determined by the
    arc-length curvature minimization

    .. math::

        \\alpha^{\\ast} = \\arg \\min_{\\alpha} \\int_{-1}^1
        \\frac{\\vert y''_{\\alpha}(x)\\vert^2}
        {(1 + \\vert y'_{\\alpha}(x) \\vert^2)^{\\frac{5}{2}}}
        \\mathrm{d} x.

    **Boundary Conditions:**

    The following boundary conditions are added to the data :math:`(x, y)`:

    * If ``func_type`` is `1`, then the point :math:`(-1, 0)` is imposed to the
      regression.
    * If ``func_type`` is `2`, no boundary condition is added to the data.

    **Interpolation Points:**

    The best practice is to provide an array of interpolation points that are
    equally distanced on the logarithmic scale. For instance, to produce four
    interpolation points in the interval :math:`[10^{-2}, 1]`:

    .. code-block:: python

        >>> import numpy
        >>> ti = numpy.logspace(-2, 1, 4)

    If ``ti`` is an integer, then `ti` number of interpolation points are
    automatically generated on Chebyshev nodes of degree `ti`, which are
    defined by

    .. math::

        x_i = \\cos \\left( \\frac{(2i+1)\\pi}{2n}  \\right)

    where :math:`i=0, \\dots, n-1` and :math:`n` is the integer number ``ti``.

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
    :math:`\\mathbf{A} + t \\mathbf{B}` using ``crf`` algorithm and the
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
        >>> f = InterpolateSchatten(A, B, p=2, kind='crf', ti=ti, func_type=1,
        ...                         scale=None)

        >>> # Interpolate at an inquiry point t = 0.4
        >>> t = 4e-1
        >>> f(t)
        1.737489512386539

    Alternatively, call :meth:`imate.InterpolateSchatten.interpolate` to
    interpolate at points `t`:

    .. code-block:: python

        >>> # This is the same as f(t)
        >>> f.interpolate(t)
        1.737489512386539

    Since we set ``scale=None``, the scale parameter :math:`\\alpha` is
    determined optimally. This parameter can be accessed by

    .. code-block:: python

        >>> f.get_scale()
        0.17949219

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
        >>> f = InterpolateSchatten(A, B, p=2, options=options, kind='crf',
        ...                         ti=ti)
        >>> f(t)
        1.7015559499058706

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
        >>> f = InterpolateSchatten(A, B, kind='crf', ti=ti)

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

    .. image:: ../_static/images/plots/interpolate_schatten_crf.png
        :align: center
        :class: custom-dark

    From the error plot in the above, it can be seen that with only four
    interpolation points, the error of interpolation for a wide range of
    :math:`t` is no more than :math:`0.003 \\%`. Also, note that the error on
    the interpolant points :math:`t_i=[10^{-2}, 10^{-1}, 1, 10]` is zero since
    the interpolation scheme honors the exact function value at the
    interpolation points.
    """

    # ====
    # Init
    # ====

    def __init__(self, A, B=None, p=0, options={}, verbose=False, ti=[],
                 scale=None, func_type=1):
        """
        Initializes the base class and attributes.
        """

        if (ti is None) or (ti == []):
            raise ValueError('"ti" should be an integer, list, or array.')
        elif numpy.isscalar(ti):
            if not isinstance(ti, (int, numpy.integer)):
                raise ValueError('"ti" as a scalar should be an integer.')

            # Produce ti number of Chebyshev nodes
            degree = ti
            ti = ChebyshevRationalFunctionsMethod.chebyshev_rational_roots(
                    degree)
            self.q = ti.size

        # Base class constructor. This will compute self.tau0 and self.t_i
        super(ChebyshevRationalFunctionsMethod, self).__init__(
                A, B=B, p=p, ti=ti, options=options, verbose=verbose)

        # Initialize attributes
        self.func_type = func_type
        self.interp_obj = None
        self.scale = scale
        self.coeff = None

        # Initialize interpolator
        self.initialize_interpolator()

    # ========================
    # Chebyshev rational roots
    # ========================

    @staticmethod
    def chebyshev_rational_roots(degree):
        """
        Returns Chebyshev points for Chebyshev rational function.
        """

        # Chebyshev roots. x is an array of size degree.
        x = scipy.special.roots_chebyt(degree)[0]

        # k = numpy.arange(degree). The above is equivalent to the following
        # x = numpy.cos((2*k+1)*numpy.pi/(2*degree))

        # k = numpy.arange(1, degree)
        # x = -numpy.cos(k*numpy.pi/n))

        x = numpy.sort(x)

        # Chebyshev rational roots. Convert x = (t-1)/(t+1) to t
        t = (1.0+x) / (1.0-x)

        return t

    # =============
    # linear system
    # =============

    def linear_system(self, scale, t_i, yi):
        """
        Creates a linear system T w = yi, where T_{ij} = T_[j](x_i) and
        x_i = (t_i - scale) / (t_i + scale).

        * T is (q, q) matrix
        * yi and ti are of size (q,)
        * coefficients w is of the size (q,).
        """

        scale = numpy.abs(scale)

        t_i_ = t_i / scale
        xi = (t_i_ - 1.0) / (t_i_ + 1.0)

        if self.func_type == 1:
            xi = numpy.r_[-1.0, xi]
            yi = numpy.r_[0.0, yi]

        T = numpy.zeros((xi.size, xi.size), dtype=float)
        for i in range(xi.size):
            for j in range(xi.size):
                T[i, j] = 0.5 * (1.0 - eval_chebyt(
                            j+1, xi[i]))

        Tinv = numpy.linalg.inv(T)
        coeff = numpy.dot(Tinv, yi)

        return coeff

    # ==================
    # curvature integral
    # ==================

    def curvature_integral(self, scale, t_i, yi):
        """
        Computes the arc integral of the curvature squared.

        The function f is defined by

            f(x) = sum_{i=1}^q w_i * Ti(x)

        where the coefficients w_i are solved by fitting f(x_i) = yi, where
        x_i = (t_i - scale) / (ti + scale)

        The arc curvature square integral is

            int_{-1}^{1} (f'')**2 / (1 + (f')**2)**(2.5) dx.
        """

        # ---
        # d1T
        # ---

        def d1T(n, x):
            """
            First derivative of Chebyshev polynomial of order n.
            """

            dT_dx = n * eval_chebyu(n-1, x)
            return dT_dx

        # ---
        # d2T
        # ---

        def d2T(n, x):
            """
            Second derivative of Chebyshev polynomial of order n.

            Second derivative of Chebyshev T_{n}(x) function is
            n * (T_{n}(x) - x*U_{n-1}(x)) / (x**2-1) where U is the Chebyshev
            function of the second kind.
            """

            if x == 1.0:
                d2T_dx2 = (n**4 - n**2) / 3.0
            elif x == -1.0:
                d2T_dx2 = (-1)**n * (n**4 - n**2) / 3.0
            else:
                d2T_dx2 = n * \
                        (n * eval_chebyt(n, x) - x * eval_chebyu(n-1, x)) / \
                        (x**2 - 1.0)

            return d2T_dx2

        # ---------
        # curvature
        # ---------

        def arc_curvature_squared(x, coeff):
            """
            Returns the arc curvature squared of the function:

                f(x) = sum_{i=1}^q w_i * Ti(x)

            at point x. T_i is the Chebyshev polynomial of order i. The arc
            curvature square is

                (f'')**2 / (1 + (f')**2)**(2.5)
            """

            d1_poly = 0
            for j in range(coeff.size):
                d1_poly += 0.5 * coeff[j] * d1T(j+1, x)

            d2_poly = 0
            for j in range(coeff.size):
                d2_poly += 0.5 * coeff[j] * d2T(j+1, x)

            # Arc curvature squared
            curv = d2_poly**2 / (1.0 + d1_poly**2)**(2.5)

            # Pure curvature squared
            # curv = d2_poly**2

            return curv

        # ---------

        if scale < 1e-5:
            return numpy.inf

        coeff = self.linear_system(scale, t_i, yi)
        tot_curv = quad(arc_curvature_squared, -1, 1, args=(coeff))[0]

        return tot_curv

    # =======================
    # initialize interpolator
    # =======================

    def initialize_interpolator(self):
        """
        Finds the coefficients of the interpolating function.
        """

        if self.verbose:
            print('Initialize Interpolator ...')

        if self.func_type == 1:
            yi = self.tau_i / (self.tau0 + self.t_i) - 1.0
        elif self.func_type == 2:
            yi = (self.tau_i - self.tau0) / self.t_i - 1.0
        else:
            raise ValueError('"type" should be 1 or 2.')

        if self.scale is not None:

            # Do not find optimal scale. Use the given value.
            self.coeff = self.linear_system(self.scale, self.t_i, yi)

        else:

            # Find optimal value of scale
            # init_scale = self.tau0
            init_scale = 1.0
            t_i = self.t_i
            res = scipy.optimize.minimize(
                    self.curvature_integral, init_scale, args=(t_i, yi),
                    method='Nelder-Mead')
            self.scale = numpy.abs(res.x)

            # With the optimized scale, now find coeff w_i
            self.coeff = self.linear_system(self.scale, self.t_i, yi)

        if self.verbose:
            print('tau0: %f' % self.tau0)
            print('scale: %f' % self.scale)

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
        """

        t_ = t / self.scale
        x = (t_ - 1.0) / (t_ + 1.0)

        y = numpy.zeros_like(x)
        for j in range(self.coeff.size):
            y += self.coeff[j] * 0.5 * (1.0 - eval_chebyt(j+1, x))

        if self.func_type == 1:
            tau = (y+1.0)*(self.tau0 + t)
        elif self.func_type == 2:
            tau = (y+1.0)*t + self.tau0
        else:
            raise ValueError('"type" should be 1 or 2.')

        # If input is a number, make output also a number, not an array
        if isinstance(t, numbers.Number):
            if not isinstance(tau, numbers.Number):
                tau = tau[0]

        schatten = tau * self.schatten_B
        return schatten
