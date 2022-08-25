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
from numbers import Number


# ===============================
# Monomial Basis Functions Method
# ===============================

class MonomialBasisFunctionsMethod(InterpolantBase):
    """
    Computes the trace of inverse of an invertible matrix :math:`\\mathbf{A} +
    t \\mathbf{B}` using

    .. math::

        \\frac{1}{(\\tau(t))^{q+1}} \\approx \\frac{1}{(\\tau_0)^{q+1}} +
        \\sum_{i=1}^{q+1} w_i t^i,

    where

    .. math::

        \\tau(t) = \\frac{\\mathrm{trace}\\left( (\\mathbf{A} +
        t \\mathbf{B})^{-1} \\right)}{\\mathrm{trace \\left( \\mathbf{B}^{-1}
        \\right)}}

    and :math:`\\tau_0 = \\tau(0)` and :math:`w_{q+1} = 1`.
    To find the weight coefficient :math:`w_1`, the trace is computed at the
    given interpolant point :math:`t_1` (see ``ti`` argument).

    When :math:`q = 1`, meaning that there is only one interpolant point
    :math:`t_1` with the function value  :math:`\\tau_1 = \\tau(t_1)`, the
    weight coefficient :math:`w_1` can be solved easily. In this case, the
    interpolation function becomes

    .. math::


        \\frac{1}{(\\tau(t))^2} \\approx  \\frac{1}{\\tau_0^2} + t^2 +
        \\left( \\frac{1}{\\tau_1^2} - \\frac{1}{\\tau_0^2} - t_1^2 \\right)
        \\frac{t}{t_1}.

    .. note::

        This class accepts only *one* interpolant point (:math:`q = 1`). That
        is, the argument ``ti`` should be only one number or a
        list of the length 1.

    **Class Inheritance:**

    .. inheritance-diagram::
        imate.InterpolateTraceinv.MonomialBasisFunctionsMethod
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

    :example:

    This class can be invoked from
    :class:`imate.InterpolateTraceinv.InterpolateTraceinv` module
    using ``method='MBF'`` argument.

    .. code-block:: python

        >>> from imate import generate_matrix
        >>> from imate import InterpolateTraceinv

        >>> # Create a symmetric positive-definite matrix, size (20**2, 20**2)
        >>> A = generate_matrix(size=20)

        >>> # Create an object that interpolates trace of inverse of A+tI
        >>> # where I is identity matrix.
        >>> TI = InterpolateTraceinv(A, method='MBF')

        >>> # Interpolate A+tI at some input point t
        >>> t = 4e-1
        >>> trace = TI.interpolate(t)

    .. seealso::

        This class can only accept one interpolant point. A better method is
        ``'RMBF'`` which accepts arbitrary number of interpolant points. It is
        recommended to use the ``'RMBF'`` (see
        :class:`imate.InterpolateTraceinv.RootMonomialBasisFunctionsMethod`)
        instead of this class.
    """

    # ====
    # Init
    # ====

    def __init__(self, A, B=None, p=0, ti=[], options={},
                 verbose=False):
        """
        Initializes the base class and attributes, namely, the trace at the
        interpolant point.
        """

        # Base class constructor
        super(MonomialBasisFunctionsMethod, self).__init__(
                A, B=B, p=p, ti=ti, options=options, verbose=verbose)

        # t1
        if ti == []:
            self.t1 = 1.0 / self.tau0
        else:
            # Check number of interpolant points
            if not isinstance(ti, Number):
                raise TypeError("'ti' for the 'MBF' method should be a ' + \
                                'single number, not an array or a list of ' + \
                                'numbers.")

            self.t1 = ti

        # Initialize interpolator
        self.tau1 = None
        self.initialize_interpolator()

        # Attributes
        self.t_i = numpy.array([self.t1])
        self.q = self.t_i.size

    # =======================
    # Initialize Interpolator
    # =======================

    def initialize_interpolator(self):
        """
        Computes the trace at the interpolant point. This function is used
        internally.
        """

        if self.verbose:
            print('Initialize interpolator ...')

        schatten_1 = self.eval(self.t1)
        self.tau1 = schatten_1 / self.schatten_B

        if self.verbose:
            print('Done.')

    # ===========
    # interpolate
    # ===========

    def interpolate(self, t):
        """
        Interpolates :math:`\\mathrm{trace} \\left( (\\mathbf{A} + t
        \\mathbf{B})^{-1} \\right)` at :math:`t`.

        This is the main interface function of this module and it is used after
        the interpolation
        object is initialized.

        :param t: The inquiry point(s).
        :type t: float, list, or numpy.array

        :return: The interpolated value of the trace.
        :rtype: float or numpy.array
        """

        # Interpolate
        tau = numpy.sqrt(t**2 + (self.tau1**2 - self.tau0**2 -
                         self.t1**2)*(t/self.t1) + self.tau0**2)

        schatten = tau * self.schatten_B
        return schatten
