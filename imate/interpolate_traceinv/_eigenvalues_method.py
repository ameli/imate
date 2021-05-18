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
    Computes the trace of inverse of an invertible matrix :math:`\\mathbf{A} +
    t \\mathbf{B}` using eigenvalues of :math:`\\mathbf{A}`  and
    :math:`\\mathbf{B}`.

    The trace of computed by

    .. math::

        \\mathrm{trace}\\left( (\\mathbf{A} + t \\mathbf{B})^{-1} \\right)
        = \\sum_{i = 1}^n \\frac{1}{\\lambda_i + t \\mu_i}

    where :math:`\\lambda_i` is the eigenvalue of :math:`\\mathbf{A}`
    and :math:`\\mu_i` is the eigenvalue of :math:`\\mathbf{B}`.
    This class does not accept interpolant points as the result is not
    interpolated.

    **Class Inheritance:**

    .. inheritance-diagram:: imate.InterpolateTraceinv.EigenvaluesMethod
        :parts: 1

    :param A: Invertible matrix, can be either dense or sparse matrix.
    :type A: numpy.ndarray

    :param B: Invertible matrix, can be either dense or sparse matrix.
    :type B: numpy.ndarray

    :param traceinv_options: A dictionary of input arguments for
        :mod:`imate.traceinv.traceinv` module.
    :type traceinv_options: dict

    :param non_zero_ratio: The ratio of the number of eigenvalues to be assumed
        non-zero over all eigenvalues. This option is only used for sparse
        matrices where as assume some of eigenvalues are very small and we are
        only interested in computing non-zero eigenvalues. In practice, it is
        not possible to compute all eigenvalues of a large sparse matrix.
        Default is ``0.9`` indicating to compute 90 percent of the eigenvalues
        with the largest magnitude and assume the rest of the eigenvalues are
        zero.
    :type non_zero_ratio: int

    :param tol: tol of computing eigenvalues. This option is only
        used for sparse matrices. Default value is ``1e-3``.
    :type tol: float

    :param verbose: If ``True``, prints some information on the computation
        process. Default is ``False``.
    :type verbose: bool

    .. note::

        The input matrices :math:`\\mathbf{A}` and :math:`\\mathbf{B}` can be
        either sparse or dense. In case of a **sparse matrix**, only some of
        the eigenvalues with the largest magnitude is computed  and the rest of
        its eigenvalues is assumed to be negligible. The ratio of computed
        eigenvalues over the total number of eigenvalues can be set by
        ``non_zero_ratio``. The tolerance at which the eigenvalues are computed
        can be set by ``tol``.

    **Example:**

    This class can be invoked from
    :class:`imate.InterpolateTraceinv.InterpolateTraceinv` module using
    ``method='EIG'`` argument.

    .. code-block:: python

        >>> from imate import generate_matrix
        >>> from imate import InterpolateTraceinv

        >>> # Create a symmetric positive-definite matrix, size (20**2, 20**2)
        >>> A = generate_matrix(size=20)

        >>> # Create an object that interpolates trace of inverse of A+tI
        >>> # where I is identity matrix.
        >>> TI = InterpolateTraceinv(A, InterpolatiionMethod='EIG')

        >>> # Interpolate A+tI at some input point t
        >>> t = 4e-1
        >>> trace = TI.interpolate(t)

    .. seealso::

        The result of the ``EIG`` method is identical with the exact method
        ``EXT``, which is given by
        :class:`imate.InterpolateTraceinv.ExactMethod`.
    """

    # ====
    # Init
    # ====

    def __init__(self, A, B=None, traceinv_options={}, non_zero_ratio=0.9,
                 tol=1e-3, verbose=False):
        """
        Constructor of the class, which initializes the bases class and
        computes eigenvalues of the input matrices.
        """

        # Base class constructor
        super(EigenvaluesMethod, self).__init__(
                A, B=B, traceinv_options=traceinv_options, verbose=verbose)

        # Attributes
        self.non_zero_ratio = non_zero_ratio
        self.tol = tol
        self.p = 0

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
        trace = numpy.sum(1.0/(self.A_eigenvalues + t * self.B_eigenvalues))

        return trace
