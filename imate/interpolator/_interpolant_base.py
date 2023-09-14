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

from ..schatten import schatten
import numpy
import scipy
from numbers import Number


# ================
# Interpolant Base
# ================

class InterpolantBase(object):
    """
    This is the base class for the interpolation methods.

    .. inheritance-diagram::
        imate.InterpolateTraceinv.ExactMethod
        imate.InterpolateTraceinv.EigenvaluesMethod
        imate.InterpolateTraceinv.MonomialBasisFunctionsMethod
        imate.InterpolateTraceinv.InverseMonomialBasisFunctionsMethod
        imate.InterpolateTraceinv.RadialBasisFunctionsMethod
        imate.InterpolateTraceinv.RationalPolynomialFunctionsMethod
        :parts: 1

    :param A: A positive-definite matrix. Matrix can be dense or sparse.
    :type A: numpy.ndarray or scipy.sparse.csc_matrix

    :param B: A positive-definite matrix. Matrix can be dense or sparse.
        If ``None`` or not provided, it is assumed that ``B`` is an identity
        matrix of the shape of ``A``.
    :type B: numpy.ndarray or scipy.sparse.csc_matrix

    :param ti: A list or an array of points that the
        interpolator use to interpolate. The trace of inverse is computed for
        the interpolant points with exact method.
    :type ti: list(float) or numpy.array(float)

    :param method: One of the methods ``'EXT'``, ``'EIG'``, ``'MBF'``,
        ``'RMBF'``, ``'RBF'``, and ``'RPF'``. Default is ``'RMBF'``.
    :type method: string

    :param options: A dictionary of arguments to pass to
        :mod:`imate.traceinv` module.
    :type options: dict

    :param verbose: If ``True``, prints some information on the computation
        process. Default is ``False``.
    :type verbose: bool
    """

    # ====
    # Init
    # ====

    def __init__(self, A, B=None, p=2, ti=None, options={},
                 verbose=False):
        """
        The initialization function does the followings:

        * Initializes the interpolant points and the input matrix.
        * Scale the interpolant points if needed.
        * Computes the trace of inverse at the interpolant points.
        """

        # Attributes
        self.verbose = verbose

        # Matrix A
        self.A = A
        self.n = self.A.shape[0]

        # Determine to use sparse
        self.use_sparse = False
        if scipy.sparse.isspmatrix(A):
            self.use_sparse = True

        # Matrix B
        if B is not None:
            self.B = B
            self.B_is_identity = False
        else:
            # Assume B is identity matrix
            self.B_is_identity = True

            if self.use_sparse:
                # Create sparse identity matrix
                self.B = scipy.sparse.eye(self.n, format='csc')
            else:
                # Create dense identity matrix
                self.B = numpy.eye(self.n)

        # Power p of (A+tB)**p
        self.p = p

        # Compute trace at interpolant points
        self.schatten_A = None
        self.schatten_B = None
        self.tau0 = None
        self.t_i = None
        self.q = None
        self.schatten_i = None
        self.tau_i = None
        self.scale_t = None
        self.options = options

        if ti is not None:

            # Compute self.schatten_A, self.schatten_B, and self.tau0
            self._compute_schatten_of_input_matrices()

            if list(ti) != []:

                # Compute schatten norm at interpolant points
                if isinstance(ti, Number):
                    ti = [ti]
                self.t_i = numpy.array(ti)
                self.q = self.t_i.size
                self.schatten_i = self._compute_for_array(self.t_i)
                self.tau_i = self.schatten_i / self.schatten_B

                # Scale interpolant points
                self.scale_t = self.find_scale_ti()

    # ================
    # compute schatten
    # ================

    def _compute_schatten(self, A, p):
        """
        Computes Schatten norm of A, defined as follows:

        * if p is zero, computes det(A+tB)**(1/n)
        * If p is not zero, computes (trace(A+tB)**(p)/n)**(1/p)
        """

        schatten_ = schatten(A, p=self.p, **self.options)
        return schatten_

    # ================================
    # compute tracep of input matrices
    # ================================

    def _compute_schatten_of_input_matrices(self):
        """
        Computes Schatten norm of A, B, and its ratio.

        Schatten norm is defined as:
        * if p is zero, computes det(A+tB)**(1/n)
        * If p is not zero, computes (trace(A+tB)**(p)/n)**(1/p)
        """

        # trace of Ainv
        self.schatten_A = self._compute_schatten(self.A, self.p)

        # trace of Binv
        if self.B_is_identity:
            self.schatten_B = 1.0
        else:
            self.schatten_B = self._compute_schatten(self.B, self.p)

        # tau0
        self.tau0 = self.schatten_A / self.schatten_B

    # ========================
    # scale interpolant points
    # ========================

    def find_scale_ti(self):
        """
        Rescales the range of interpolant points. This function is intended to
        be used internally.

        If the largest interpolant point in ``self.ti`` is
        greater than ``1``, this function rescales their range to max at ``1``.
        The rescale is necessary if the method of interpolation is based on the
        orthogonal basis functions, which they are define to be orthogonal
        in the range :math:`t \\in [0, 1]`.
        """

        # Scale t, if some of t_i are greater than 1
        scale_t = 1.0
        if self.t_i.size > 0:
            if numpy.max(self.t_i) > 1.0:
                scale_t = numpy.max(self.t_i)

        return scale_t

    # ====
    # eval
    # ====

    def eval(self, t):
        """
        Exact solution without interpolation. This is an interface function.

        * if p is zero, computes det(A+tB)**(1/n)
        * If p is not zero,, computes (trace(A+tB)**(p))**(1/p)
        """

        schatten = self._compute(t)
        return schatten

    # =======
    # compute
    # =======

    def _compute(self, t):
        """
        Computes Schatten norm of A+tB without interpolation for one t.

        Schatten norm is defined as:
        * if p is zero, computes det(A+tB)**(1/n)
        * If p is not zero, computes (trace(A+tB)**(p)/n)**(1/p)
        """

        An = self.A + t * self.B
        schatten = self._compute_schatten(An, self.p)

        return schatten

    # =================
    # compute for array
    # =================

    def _compute_for_array(self, t_i):
        """
        Computes Schatten norm of A+tB without interpolation for an array of t.

        Schatten norm is defined as:
        * if p is zero, computes det(A+tB)**(1/n)
        * If p is not zero, computes (trace(A+tB)**(p)/n)**(1/p)
        """

        if self.verbose:
            print('Evaluate function at interpolant points ...', end='')

        if numpy.isscalar(t_i):

            # Compute for scalar input
            trace_i = self._compute(t_i)

        else:
            # Compute for an array
            trace_i = numpy.zeros(self.q)
            for i in range(self.q):
                trace_i[i] = self._compute(t_i[i])

        if self.verbose:
            print(' Done.')

        return trace_i

    # ===========
    # upper bound
    # ===========

    def upper_bound(self, t):
        """
        Upper bound of the function :math:`t \\mapsto \\mathrm{trace} \\left(
        (\\mathbf{A} + t \\mathbf{B})^{-1} \\right)`.

        The lower bound is given by

        .. math::

                \\mathrm{trace}((\\mathbf{A}+t\\mathbf{B})^{-1}) \\geq
                \\frac{n^2}{\\mathrm{trace}(\\mathbf{A}) +
                \\mathrm{trace}(t \\mathbf{B})}

        :param t: Inquiry points
        :type t: float or numpy.ndarray

        :return: Lower bound of the trace of inverse at inquiry points.
        :rtype: float or numpy.ndarray
        """

        if self.p != -1:
            raise ValueError('Lower bound is only available for "p=-1".')

        # Trace of A and B
        trace_A = numpy.trace(self.A)
        trace_B = numpy.trace(self.B)

        # Lower bound of trace of A+tB
        trace_ub = (self.n**2)/(trace_A + t*trace_B)
        schatten_ub = (trace_ub / self.n)**(1.0/self.p)

        return schatten_ub

    # =====
    # bound
    # =====

    def bound(self, t):
        """
        Lower bound of the function :math:`t \\mapsto \\mathrm{trace} \\left(
        (\\mathbf{A} + t \\mathbf{B})^{-1} \\right)`.

        The upper bound is given by

        .. math::

                \\frac{1}{\\tau(t)} \\geq \\frac{1}{\\tau_0} + t

        where

        .. math::

                \\tau(t) = \\frac{\\mathrm{trace}\\left( (\\mathbf{A} +
                t \\mathbf{B})^{-1}
                \\right)}{\\mathrm{trace}(\\mathbf{B}^{-1})}

        and :math:`\\tau_0 = \\tau(0)`.

        :param t: Inquiry points
        :type t: float or numpy.ndarray

        :return: Upper bound of the trace of inverse at inquiry points.
        :rtype: float or numpy.ndarray
        """

        # Compute trace at interpolant points
        if self.tau0 is None:
            self._compute_schatten_of_input_matrices()

        # Upper bound of schatten norm of A+tB
        tau_bound = self.tau0 + t
        schatten_bound = tau_bound * self.schatten_B

        return schatten_bound
