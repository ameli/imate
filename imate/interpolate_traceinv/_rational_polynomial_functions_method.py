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


# ====================================
# Rational Polynomial Functions Method
# ====================================

class RationalPolynomialFunctionsMethod(InterpolantBase):
    """
    Computes the trace of inverse of an invertible matrix :math:`\\mathbf{A} +
    t \\mathbf{B}` using an interpolation scheme based on rational polynomial
    functions (see details below).

    **Class Inheritance:**

    .. inheritance-diagram::
    imate.InterpolateTraceinv.RationalPolynomialFunctionsMethod
        :parts: 1

    :param A: Invertible matrix, can be either dense or sparse matrix.
    :type A: numpy.ndarray

    :param B: Invertible matrix, can be either dense or sparse matrix.
    :type B: numpy.ndarray

    :param traceinv_options: A dictionary of input arguments for
        :mod:`imate.traceinv.traceinv` module.
    :type traceinv_options: dict

    :param verbose: If ``True``, prints some information on the computation
        process. Default is ``False``.
    :type verbose: bool

    **Interpolation Method**

    Define the function

    .. math::

        \\tau(t) = \\frac{\\mathrm{trace}\\left( (\\mathbf{A} +
        t \\mathbf{B})^{-1} \\right)}{\\mathrm{trace}(\\mathbf{B}^{-1})}

    and :math:`\\tau_0 = \\tau(0)`. Then, we approximate :math:`\\tau^{-1}(t)`
    by

    .. math::

        \\tau(t) \\approx \\frac{t^p + a_{p-1} t^{p-1} + \\cdots + a_1 t +
        a_0}{t^{p+1} + b_p t^p + \\cdots + b_1 t + b_0}

    where :math:`a_0 = b_0 \\tau_0`. The rest of coefficients are found by
    solving a linear system using the function value at the interpolant points
    :math:`\\tau_i = \\tau(t_i)`.

    .. note::

        The number of interpolant points :math:`p` in this module can only be
        either :math:`p = 2` or :math:`p = 4`.

    **Example**

    This class can be invoked from
    :class:`imate.InterpolateTraceinv.InterpolateTraceinv` module
    using ``method='RPF'`` argument.

    .. code-block:: python

        >>> from imate import generate_matrix
        >>> from imate import InterpolateTraceinv

        >>> # Create a symmetric positive-definite matrix, size (20**2, 20**2)
        >>> A = generate_matrix(size=20)

        >>> # Create an object that interpolates trace of inverse of A+tI
        >>> # where I is identity matrix.
        >>> TI = InterpolateTraceinv(A, method='RPF')

        >>> # Interpolate A+tI at some input point t
        >>> t = 4e-1
        >>> trace = TI.interpolate(t)
    """

    # ====
    # Init
    # ====

    def __init__(self, A, B=None, interpolant_points=None, traceinv_options={},
                 verbose=False):
        """
        Initializes the base class and attributes.
        """

        # Base class constructor
        super(RationalPolynomialFunctionsMethod, self).__init__(
                A, B=B, interpolant_points=interpolant_points,
                traceinv_options=traceinv_options, verbose=verbose)

        # Initialize interpolator
        self.numerator = None
        self.denominator = None
        self.initialize_interpolator()

    # ================
    # Rational Poly 12
    # ================

    def rational_poly_12(self, t_i, tau_i, tau0):
        """
        Finds the coefficients :math:`(a_0, b_1, b_0)` of a Rational polynomial
        of order 1 over 2,

        .. math::

            \\tau(t) \\approx \\frac{t + a_0}{t^2 + b_1 t + b_0}

        This is used when the number of interpolant points is :math:`p = 2`.

        :param t_i: Inquiry point.
        :type t_i: float

        :param tau_i: The function value at the inquiry point :math:`t`
        :type tau_i: float

        :param tau_0: The function value at :math:`t_0 = 0`.
        :type tau_0: float
        """

        # Matrix of coefficients
        C = numpy.array([
            [t_i[0], 1-tau0/tau_i[0]],
            [t_i[1], 1-tau0/tau_i[1]]])

        # Vector of right hand side
        c = numpy.array([
            t_i[0]/tau_i[0]-t_i[0]**2,
            t_i[1]/tau_i[1]-t_i[1]**2])

        # Condition number
        if self.verbose:
            print('Condition number: %0.2e' % (numpy.linalg.cond(C)))

        # Solve with least square. NOTE: don't solve with numpy.linalg.solve.
        b = numpy.linalg.solve(C, c)
        b0 = b[1]
        b1 = b[0]
        a0 = b0*tau0

        # Output
        numerator = [1, a0]
        denominator = [1, b1, b0]

        # Check poles
        Poles = numpy.roots(denominator)
        if numpy.any(Poles > 0):
            print('denominator poles:')
            print(Poles)
            raise ValueError('rational_polynomial has positive poles.')

        return numerator, denominator

    # ================
    # rational poly 23
    # ================

    def rational_poly_23(self, t_i, tau_i, tau0):
        """
        Finds the coefficients :math:`(a_1, a_0, b_2, b_1, b_0)` of a Rational
        polynomial of order 2 over 3,

        .. math::

            \\tau(t) \\approx \\frac{t^2 + a_{1} t + a_0}{t^3 + b_2 t^2 + b_1 t
            + b_0}

        This is used when the number of interpolant points is :math:`p = 4`.

        :param t_i: Inquiry point.
        :type t_i: float

        :param tau_i: The function value at the inquiry point :math:`t`
        :type tau_i: float

        :param tau_0: The function value at :math:`t_0 = 0`.
        :type tau_0: float
        """

        # Matrix of coefficients
        C = numpy.array([
            [t_i[0]**2, t_i[0], 1-tau0/tau_i[0], -t_i[0]/tau_i[0]],
            [t_i[1]**2, t_i[1], 1-tau0/tau_i[1], -t_i[1]/tau_i[1]],
            [t_i[2]**2, t_i[2], 1-tau0/tau_i[2], -t_i[2]/tau_i[2]],
            [t_i[3]**2, t_i[3], 1-tau0/tau_i[3], -t_i[3]/tau_i[3]]])

        # Vector of right hand side
        c = numpy.array([
            t_i[0]**2/tau_i[0]-t_i[0]**3,
            t_i[1]**2/tau_i[1]-t_i[1]**3,
            t_i[2]**2/tau_i[2]-t_i[2]**3,
            t_i[3]**2/tau_i[3]-t_i[3]**3])

        # Condition number
        if self.verbose:
            print('Condition number: %0.2e' % (numpy.linalg.cond(C)))

        # Solve with least square. NOTE: don't solve with numpy.linalg.solve.
        b = numpy.linalg.solve(C, c)
        b2 = b[0]
        b1 = b[1]
        b0 = b[2]
        a1 = b[3]
        a0 = b0*tau0

        # Output
        numerator = [1, a1, a0]
        denominator = [1, b2, b1, b0]

        # Check poles
        Poles = numpy.roots(denominator)
        if numpy.any(Poles > 0):
            print('denominator poles:')
            print(Poles)
            raise ValueError('rational_polynomial has positive poles.')

        return numerator, denominator

    # =============
    # rational poly
    # =============

    def rational_poly(self, t, numerator, denominator):
        """
        Evaluates rational polynomial.

        :param t: Inquiry point.
        :type t: float

        :param numerator: A list of coefficients of polynomial in the numerator
            of the rational polynomial function.
        :type numerator: list

        :param denominator: A list of coefficients of polynomial in the
            denominator of the rational polynomial function.
        :type numerator: list
        """

        return numpy.polyval(numerator, t) / numpy.polyval(denominator, t)

    # =======================
    # initialize interpolator
    # =======================

    def initialize_interpolator(self):
        """
        Sets ``self.numerator`` and ``self.denominator`` which are list of the
        coefficients of the numerator and denominator of the rational
        polynomial.
        """

        if self.verbose:
            print('Initialize interpolator ...')

        # Coefficients of a linear system
        if self.p == 2:
            self.numerator, self.denominator = self.rational_poly_12(
                    self.t_i, self.tau_i, self.tau0)

        elif self.p == 4:
            self.numerator, self.denominator = self.rational_poly_23(
                    self.t_i, self.tau_i, self.tau0)

        else:
            raise ValueError('In rational_polynomial method, the number of ' +
                             'interpolant points, p, should be 2 or 4.')

        if self.verbose:
            print('Done.')

    # ===========
    # interpolate
    # ===========

    def interpolate(self, t):
        """
        Interpolates :math:`\\mathrm{trace} \\left( (\\mathbf{A} +
        t \\mathbf{B})^{-1} \\right)` at :math:`t`.

        This is the main interface function of this module and it is used after
        the interpolation object is initialized.

        :param t: The inquiry point(s).
        :type t: float, list, or numpy.array

        :return: The interpolated value of the trace.
        :rtype: float or numpy.array
        """

        tau = self.rational_poly(t, self.numerator, self.denominator)
        trace = tau*self.trace_Binv

        return trace
