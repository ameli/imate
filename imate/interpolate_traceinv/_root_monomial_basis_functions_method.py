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
# Root Monomial Basis Functions Method
# ====================================

class RootMonomialBasisFunctionsMethod(InterpolantBase):
    """
    Computes the trace of inverse of an invertible matrix :math:`\\mathbf{A} +
    t \\mathbf{B}` using an interpolation scheme based on root monomial basis
    functions (see details below).

    **Class Inheritance:**

    .. inheritance-diagram::
        imate.InterpolateTraceinv.RootMonomialBasisFunctionsMethod
        :parts: 1

    :param A: Invertible matrix, can be either dense or sparse matrix.
    :type A: numpy.ndarray

    :param B: Invertible matrix, can be either dense or sparse matrix.
    :type B: numpy.ndarray

    :param traceinv_options: A dictionary of input arguments fo
        :mod:`imate.traceinv.traceinv` module.
    :type traceinv_options: dict

    :param verbose: If ``True``, prints some information on the computation
        process. Default is ``False``.
    :type verbose: bool

    :param basis_functions_type: One of the types ``'NonOrthogonal'``,
        ``'Orthogonal'`` and ``'Orthogonal2'``. Default is ``'orthogonal2'``.
    :type basisFunctionsType: string

    **Interpolation Method**

    Define the function

    .. math::

        \\tau(t) = \\frac{\\mathrm{trace}\\left( (\\mathbf{A} +
        t \\mathbf{B})^{-1} \\right)}{\\mathrm{trace}(\\mathbf{B}^{-1})}

    and :math:`\\tau_0 = \\tau(0)`. Then, we approximate :math:`\\tau^{-1}(t)`
    by

    .. math::

        \\frac{1}{\\tau(t)} \\approx \\frac{1}{\\tau_0} + \\sum_{i = 0}^p w_i
        \\phi_i(t),

    where  :math:`\\phi_i` are some known basis functions, and :math:`w_i` are
    the coefficients of the linear basis functions. The first coefficient is
    set to :math:`w_{0} = 1` and the rest of the weights are to be found from
    the known function values :math:`\\tau_i = \\tau(t_i)` at some given
    interpolant points :math:`t_i`.

    **Basis Functions:**

    In this module, three kinds of basis functions which can be set by the
    argument ``basis_functions_type``.

    When ``basis_functions_type`` is set to ``NonOrthogonal``, the basis
    functions are the root of the monomial functions defined by

    .. math::

        \\phi_i(t) = t^{\\frac{1}{i+1}}, \\qquad i = 0, \\dots, p.

    When ``basis_functions_type`` is set to ``'Orthogonal'`` or
    ``'Orthogonal2'``, the orthogonal form of the above basis functions are
    used. Orthogonal basis functions are formed by the above non-orthogonal
    functions as

    .. math::

        \\phi_i^{\\perp}(t) = \\alpha_i \\sum_{j=1}^i a_{ij} \\phi_j(t)

    The coefficients :math:`\\alpha_i` and :math:`a_{ij}` can be obtained by
    the python package `Orthogoanl Functions
    <https://ameli.github.io/Orthogonal-Functions>`_. These coefficients are
    hard-coded in this function up to :math:`i = 9`. Thus, in this module, up
    to nine interpolant points are supported.

    .. warning::

        The non-orthogonal basis functions can lead to ill-conditioned system
        of equations for finding the weight coefficients :math:`w_i`. When the
        number of interpolating points is large (such as :math:`p > 6`), it is
        recommended to use the orthogonalized set of basis functions.

    .. note::

        The recommended basis function type is ``'Orthogonal2'``.


    **Example**

    This class can be invoked from
    :class:`imate.InterpolateTraceinv.InterpolateTraceinv` module using
    ``method='RMBF'`` argument.

    .. code-block:: python

        >>> from imate import generate_matrix
        >>> from imate import InterpolateTraceinv

        >>> # Create a symmetric positive-definite matrix, size (20**2, 20**2)
        >>> A = generate_matrix(size=20)

        >>> # Create an object that interpolates trace of inverse of A+tI
        >>> # where I is identity matrix.
        >>> TI = InterpolateTraceinv(A, method='RMBF')

        >>> # Interpolate A+tI at some input point t
        >>> t = 4e-1
        >>> trace = TI.interpolate(t)

    .. seealso::

        The other class that provides interpolation with basis functions method
        is :mod:`imate.InterpolateTraceinv.Monomialbasis_functionsMethod`.
    """

    # ====
    # Init
    # ====

    def __init__(self, A, B=None, interpolant_points=None, traceinv_options={},
                 verbose=False, basis_functions_type='Orthogonal2'):
        """
        Initializes the base class and the attributes, namely, the computes the
        trace at interpolant points.
        """

        # Base class constructor
        super(RootMonomialBasisFunctionsMethod, self).__init__(
                A, B=B, interpolant_points=interpolant_points,
                traceinv_options=traceinv_options, verbose=verbose)

        self.basis_functions_type = basis_functions_type

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
        if self.basis_functions_type == 'NonOrthogonal':
            # Form a linear system for weights w
            b = (1.0/self.tau_i) - (1.0/self.tau0) - self.t_i
            C = numpy.zeros((self.p, self.p))
            for i in range(self.p):
                for j in range(self.p):
                    C[i, j] = self.basis_functions(j, self.t_i[i])

            if self.verbose:
                print('Condition number: %f' % (numpy.linalg.cond(C)))

            self.w = numpy.linalg.solve(C, b)

        elif self.basis_functions_type == 'Orthogonal':

            # Method 2: Use orthogonal basis functions
            self.alpha, self.a = self.orthogonal_basis_function_coefficients()

            if self.alpha.size < self.t_i.size:
                raise ValueError('Cannot regress order higher than %d. ' +
                                 'Decrease the number of interpolation ' +
                                 'points.' % (self.alpha.size))

            # Form a linear system Cw = b for weights w
            b = numpy.zeros(self.p+1)
            b[:-1] = (1.0/self.tau_i) - (1.0/self.tau0)
            b[-1] = 1.0
            C = numpy.zeros((self.p+1, self.p+1))
            for i in range(self.p):
                for j in range(self.p+1):
                    C[i, j] = self.basis_functions(j, self.t_i[i]/self.scale_t)

            # The coefficient of term "t" should be 1.
            C[-1, :] = self.alpha[:self.p+1]*self.a[:self.p+1, 0]

            if self.verbose:
                print('Condition number: %f' % (numpy.linalg.cond(C)))

            # Solve weights
            self.w = numpy.linalg.solve(C, b)

        elif self.basis_functions_type == 'Orthogonal2':

            # Method 3: Use orthogonal basis functions
            self.alpha, self.a = self.orthogonal_basis_function_coefficients()

            if self.alpha.size < self.t_i.size:
                raise ValueError('Cannot regress order higher than %d. ' +
                                 'Decrease the number of interpolation ' +
                                 'points.' % (self.alpha.size))

            # Form a linear system Aw = b for weights w
            b = (1.0/self.tau_i) - (1.0/self.tau0) - self.t_i
            C = numpy.zeros((self.p, self.p))
            for i in range(self.p):
                for j in range(self.p):
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

        Depending on ``basis_functions_type``, the basis functions are:

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

        if self.basis_functions_type == 'NonOrthogonal':
            return RootMonomialBasisFunctionsMethod.phi(j+2, t)

        elif self.basis_functions_type == 'Orthogonal':

            # Use Orthogonal basis functions
            alpha, a = self.orthogonal_basis_function_coefficients()

            phi_perp = 0
            for i in range(a.shape[1]):
                phi_perp += alpha[j]*a[j, i] * \
                        RootMonomialBasisFunctionsMethod.phi(i+1, t)

            return phi_perp

        elif self.basis_functions_type == 'Orthogonal2':

            # Use Orthogonal basis functions
            alpha, a = self.orthogonal_basis_function_coefficients()

            phi_perp = 0
            for i in range(a.shape[1]):
                phi_perp += alpha[j]*a[j, i] * \
                    RootMonomialBasisFunctionsMethod.phi(i+2, t)

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
        `Orthogonal Functions <https://ameli.github.io/Orthogonal-Functions>`_.

        Install this package by

            ::

                pip install OrthogonalFunctions

        * To generate the coefficients corresponding to ``Orthogonal`` basis:

          ::

            gen-orth -n 9 -s 0

        * To generate the coefficients corresponding to ``Orthogonal2`` basis:

          ::

            gen-orth -n 9 -s 1

        :return: Weight coefficients of the orthogonal basis functions.
        :rtype: numpy.array, numpy.ndarray
        """

        p = 9
        a = numpy.zeros((p, p), dtype=float)

        if self.basis_functions_type == 'Orthogonal':
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

        elif self.basis_functions_type == 'Orthogonal2':
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

        Depending on the ``basis_functions_type``, the interpolation is as
        follows:

        For ``'NonOrthogonal'`` basis:

        .. math::

            \\frac{1}{\\tau(t)} = \\frac{1}{\\tau_0} + t + \\sum_{j=1}^p w_j
            \\phi_j(t).

        For ``'Orthogonal'`` and ``'Orthogonal2'`` bases:

        .. math::

            \\frac{1}{\\tau(t)} = \\frac{1}{\\tau_0} + t + \\sum_{j=1}^p w_j
            \\phi_j(t).
        """

        if self.basis_functions_type == 'NonOrthogonal':

            S = 0.0
            for j in range(self.p):
                S += self.w[j] * self.basis_functions(j, t)
            tau = 1.0 / (1.0/self.tau0+S+t)

        elif self.basis_functions_type == 'Orthogonal':

            S = 0.0
            for j in range(self.w.size):
                S += self.w[j] * self.basis_functions(j, t/self.scale_t)
            tau = 1.0 / (1.0/self.tau0+S)

        elif self.basis_functions_type == 'Orthogonal2':

            S = 0.0
            for j in range(self.p):
                S += self.w[j] * self.basis_functions(j, t/self.scale_t)
            tau = 1.0 / (1.0/self.tau0+S+t)

        # Compute trace from tau
        trace = tau * self.trace_Binv

        return trace
