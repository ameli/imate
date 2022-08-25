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
    Computes the trace of inverse of an invertible matrix :math:`\\mathbf{A} +
    t \\mathbf{B}` using an interpolation scheme based on rational polynomial
    functions (see details below).

    **Class Inheritance:**

    .. inheritance-diagram::
        imate.InterpolateTraceinv.RadialBasisFunctionsMethod
        :parts: 1

    :param A: Invertible matrix, can be either dense or sparse matrix.
    :type A: numpy.ndarray

    :param B: Invertible matrix, can be either dense or sparse matrix.
    :type B: numpy.ndarray

    :param options: A dictionary of input arguments for
        :mod:`imate.traceinv.traceinv` module.
    :type options: dict

    :param verbose: If ``True``, prints some information on the computation
        process. Default is ``False``.
    :type verbose: bool

    :param function_type: Can be ``1``, ``2``, or ``3``, which defines
        different radial basis functions (see details below).
    :type function_type: int

    **Interpolation Method**

    Define the function

    .. math::

        \\tau(t) = \\frac{\\mathrm{trace}\\left( (\\mathbf{A} +
        t \\mathbf{B})^{-1} \\right)}{\\mathrm{trace}(\\mathbf{B}^{-1})}

    and :math:`\\tau_0 = \\tau(0)`. Then, we approximate :math:`\\tau(t)` by
    radial basis functions as follows. Define

    .. math::

        x(t) = \\log t

    Depending whether ``function_type`` is set to ``1``, ``2``, or ``3``, one
    of the following functions is defined:

    .. math::
        :nowrap:

        \\begin{eqnarray}
        y_1(t) &= \\frac{1}{\\tau(t)} - \\frac{1}{\\tau_0} - t, \\
        y_2(t) &= \\frac{\\frac{1}{\\tau(t)}}{\\frac{1}{\\tau_0} + t} - 1, \\
        y_3(t) &= 1 - \\tau(t) \\left( \\frac{1}{\\tau_0} + t \\right).
        \\end{eqnarray}

    * The set of data :math:`(x, y_1(x))` are interpolated using
      *cubic splines*.
    * The set of data :math:`(x, y_2(x))` and :math:`(x, y_3(x))` are
      interpolated using *Gaussian radial basis functions*.

    **Example**

    This class can be invoked from
    :class:`imate.InterpolateTraceinv.InterpolateTraceinv` module using
    ``method='RBF'`` argument.

    .. code-block:: python

        >>> from imate import generate_matrix
        >>> from imate import InterpolateTraceinv

        >>> # Create a symmetric positive-definite matrix, size (20**2, 20**2)
        >>> A = generate_matrix(size=20)

        >>> # Create an object that interpolates trace of inverse of A+tI
        >>> # where I is identity matrix.
        >>> TI = InterpolateTraceinv(A, method='RBF')

        >>> # Interpolate A+tI at some input point t
        >>> t = 4e-1
        >>> trace = TI.interpolate(t)
    """

    # ====
    # Init
    # ====

    def __init__(self, A, B=None, p=0, ti=[], scale=None, func_type=1,
                 options={}, verbose=False):
        """
        Initializes the base class and attributes.
        """

        if ti is None:
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

        # k = numpy.arange(degree)
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

        schatten = tau * self.schatten_B
        return schatten
