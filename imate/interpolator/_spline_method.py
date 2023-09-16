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


# =============
# Spline Method
# =============

class SplineMethod(InterpolantBase):
    """
    Interpolate Schatten norm (or anti-norm) of an affine matrix function using
    Spline (spl) method.

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
        Method of interpolation. For this class, ``kind`` is ``spl``.

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
    eval
    interpolate
    bound
    upper_bound
    plot

    Notes
    -----

    **Schatten Norm:**

    In this class, the Schatten :math:`p`-norm of the matrix
    :math:`\\mathbf{A}` is defined by

    .. math::
        :label: schatten-eq-11

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
        factor :math:`\\frac{1}{n}` in :math:numref:`schatten-eq-11`. However,
        this factor is justified by the continuity granted by

        .. math::
            :label: schatten-continuous-11

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

        x = \\frac{t - 1}{t + 1},

    Also, if ``func_type=1``, then :math:`y` is defined by

    .. math::

        y = \\frac{\\tau_p(t)}{\\tau_{p, 0} + t} - 1,

    and, if ``func_type=2``, then :math:`y` is defined by

    .. math::

        y = \\frac{\\tau_p(t) -\\tau_{p, 0}}{t} - 1.

    The spline method interpolates the data :math:`(x, y)` by cubic
    splines.

    **Boundary Conditions:**

    The following boundary conditions are added to the data :math:`(x, y)`:

    * If ``func_type`` is `1`, then the two points :math:`(-1, 0)` and
      :math:`(1, 0)` are imposed to the spline.
    * If ``func_type`` is `2`, then the point :math:`(1, 0)` is imposed to the
      spline.

    **Interpolation Points:**

    The best practice is to provide an array of interpolation points that are
    equally distanced on the logarithmic scale. For instance, to produce four
    interpolation points in the interval :math:`[10^{-2}, 1]`:

    .. code-block:: python

        >>> import numpy
        >>> ti = numpy.logspace(-2, 1, 4)

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
    :math:`\\mathbf{A} + t \\mathbf{B}` using ``spl`` algorithm and the
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
        >>> f = InterpolateSchatten(A, B, p=2, kind='spl', ti=ti, func_type=1)

        >>> # Interpolate at an inquiry point t = 0.4
        >>> t = 4e-1
        >>> f(t)
        1.7372898625858328

    Alternatively, call :meth:`imate.InterpolateSchatten.interpolate` to
    interpolate at points `t`:

    .. code-block:: python

        >>> # This is the same as f(t)
        >>> f.interpolate(t)
        1.7372898625858328

    To evaluate the exact value of the Schatten norm at point `t` without
    interpolation, call :meth:`imate.InterpolateSchatten.eval` function:

    .. code-block:: python

        >>> # This evaluates the function value at t exactly (no interpolation)
        >>> f.eval(t)
        1.7374809371539666

    It can be seen that the relative error of interpolation compared to the
    exact solution in the above is :math:`0.01 \\%` using only four
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
        >>> f = InterpolateSchatten(A, B, p=2, options=options, kind='spl',
        ...                         ti=ti)
        >>> f(t)
        1.7076411989963456

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
        >>> f = InterpolateSchatten(A, B, kind='spl', ti=ti)

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

    .. image:: ../_static/images/plots/interpolate_schatten_spl.png
        :align: center
        :class: custom-dark

    From the error plot in the above, it can be seen that with only four
    interpolation points, the error of interpolation for a wide range of
    :math:`t` is no more than :math:`0.0013 \\%`. Also, note that the error on
    the interpolant points :math:`t_i=[10^{-2}, 10^{-1}, 1, 10]` is zero since
    the interpolation scheme honors the exact function value at the
    interpolation points.
    """

    # ====
    # Init
    # ====

    def __init__(self, A, B=None, p=0, options={}, verbose=False, ti=[],
                 func_type=1):
        """
        Initializes the base class and attributes.
        """

        if (ti is None) or (ti == []):
            raise ValueError('"ti" should be a list or array.')

        # Base class constructor. This will compute self.tau0 and self.t_i
        super(SplineMethod, self).__init__(
                A, B=B, p=p, ti=ti, options=options, verbose=verbose)

        # Initialize attributes
        self.func_type = func_type
        self.interp_obj = None

        # Initialize interpolator
        self.initialize_interpolator()

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

        xi = (self.t_i - 1.0) / (self.t_i + 1.0)

        if self.func_type == 1:
            xi = numpy.r_[-1.0, xi, 1.0]
            yi = numpy.r_[0.0, yi, 0.0]
        else:
            xi = numpy.r_[xi, 1.0]
            yi = numpy.r_[yi, 0.0]

        self.interp_obj = scipy.interpolate.CubicSpline(xi, yi)

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
        the interpolation
        object is initialized.

        :param t: The inquiry point(s).
        :type t: float, list, or numpy.array

        :return: The interpolated value of the trace.
        :rtype: float or numpy.array
        """

        x = (t - 1.0) / (t + 1.0)
        y = self.interp_obj(x)

        if self.func_type == 1:
            tau = (y+1.0)*(self.tau0 + t)
        elif self.func_type == 2:
            tau = (y+1.0)*t + self.tau0
        else:
            raise ValueError('"type" should be 1 or 2.')

        schatten = tau * self.schatten_B
        return schatten
