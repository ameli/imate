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

    :param options: A dictionary of input arguments for
        :mod:`imate.traceinv.traceinv` module.
    :type options: dict

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

        \\tau(t) \\approx \\frac{t^q + a_{q-1} t^{q-1} + \\cdots + a_1 t +
        a_0}{t^{q+1} + b_p t^q + \\cdots + b_1 t + b_0}

    where :math:`a_0 = b_0 \\tau_0`. The rest of coefficients are found by
    solving a linear system using the function value at the interpolant points
    :math:`\\tau_i = \\tau(t_i)`.

    .. note::

        The number of interpolant points :math:`q` in this module can only be
        either :math:`q = 2` or :math:`q = 4`.

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

    def __init__(self, A, B=None, p=0, ti=[], options={}, verbose=False):
        """
        Initializes the base class and attributes.
        """

        if ti is None:
            raise ValueError('"ti" should be a list or array.')

        # Base class constructor
        super(RationalPolynomialFunctionsMethod, self).__init__(
                A, B=B, p=p, ti=ti, options=options, verbose=verbose)

        # Initialize interpolator
        self.numerator = None
        self.denominator = None
        self.initialize_interpolator()

    # =================
    # fit rational poly
    # =================

    def fit_rational_poly(self, t_i, tau_i, tau0):
        """
        Finds the coefficients :math:`(a_1, a_0, b_2, b_1, b_0)` of a Rational
        polynomial of order 2 over 3,

        .. math::

            \\tau(t) \\approx \\frac{t^2 + a_{1} t + a_0}{t^3 + b_2 t^2 + b_1 t
            + b_0}

        This is used when the number of interpolant points is :math:`q = 4`.

        :param t_i: Inquiry point.
        :type t_i: float

        :param tau_i: The function value at the inquiry point :math:`t`
        :type tau_i: float

        :param tau_0: The function value at :math:`t_0 = 0`.
        :type tau_0: float
        """

        # Length of t_i should be 2q+1
        q = t_i.size // 2
        if q != t_i.size/2:
            print(t_i.size)
            print(q)
            raise ValueError('In rational polynomial interpolation method, ' +
                             'the number of interpolation points should be ' +
                             'even.')

        # Matrix of coefficients
        C = numpy.zeros((2*q, 2*q), dtype=float)
        c = numpy.zeros((2*q,), dtype=float)

        for i in range(2*q):
            c[i] = tau_i[i] * t_i[i]**q - t_i[i]**(q+1)

            C[i, 0] = 1.0-tau_i[i]/tau0
            for j in range(1, q+1):
                C[i, j] = t_i[i]**j
            for j in range(1, q):
                C[i, q+j] = -tau_i[i] * t_i[i]**j

        # Condition number
        if self.verbose:
            print('Condition number: %0.2e' % (numpy.linalg.cond(C)))

        # Solve with least square. NOTE: don't solve with numpy.linalg.solve.
        x = numpy.linalg.solve(C, c)
        x = list(x)
        b = x[:q+1]
        a = x[q+1:]
        a0 = b[0]/tau0
        a = [a0] + a

        # Output
        numerator = [1] + b[::-1]
        denominator = [1] + a[::-1]

        # Check poles
        Poles = numpy.roots(denominator)
        if numpy.any(Poles > 0):
            print('denominator poles: ', end='')
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

        self.numerator, self.denominator = self.fit_rational_poly(
                self.t_i, self.tau_i, self.tau0)

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
        schatten = tau*self.schatten_B

        return schatten
