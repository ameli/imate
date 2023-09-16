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
    Interpolate Schatten norm (or anti-norm) of an affine matrix function using
    monomial basis functions (MBF) method.

    This class accepts only *one* interpolant point (:math:`q = 1`). That is,
    the argument ``ti`` should be only one number or a list of the length 1.

    A better method is ``'imbf'`` which accepts arbitrary number of interpolant
    points. It is recommended to use the ``'imbf'`` (see
    :ref:`imate.InterpolateSchatten.imbf`).

    .. note::

        If :math:`p=2`, it is recommended to use the interpolation with ``mbf``
        method since it provides the exact function values.

    Parameters
    ----------

    A : numpy.ndarray or scipy.sparse matrix
        Symmetric positive-definite matrix (positive-definite if `p` is
        non-positive). Matrix can be dense or sparse.

        .. warning::

            Symmetry and positive (semi-) definiteness of `A` will not be
            checked. Make sure `A` satisfies these conditions.

    B : numpy.ndarray or scipy.sparse matrix, default=None
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
        Interpolation point. For this class, the interpolation point should be
        a single point. If an empty list is given, `i.e.`, ``[]``, a default
        interpolant point is set as :math:`t_1 = \\tau_{p, 0}^{-1}` (see the
        definition of :math:`\\tau_p, 0` in the Notes). The interpolator honors
        the exact function values at the interpolant point.

    See Also
    --------

    imate.InterpolateTrace
    imate.InterpolateLogdet
    imate.schatten

    Attributes
    ----------

    kind : str
        Method of interpolation. For this class, ``kind`` is ``mbf``.

    verbose : bool
        Verbosity of the computation process

    n : int
        Size of the matrix

    q : int
        Number of interpolant points. For this class, `q` is `1`.

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
        :label: schatten-eq-5

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
        factor :math:`\\frac{1}{n}` in :math:numref:`schatten-eq-5`. However,
        this factor is justified by the continuity granted by

        .. math::
            :label: schatten-continuous-5

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

    The interpolator is initialized by providing one interpolant point
    :math:`t_i`. The interpolator can interpolate the above function at
    arbitrary inquiry points :math:`t \\in [t_1, t_p]` using

    .. math::

        (\\tau_p(t))^{q+1} \\approx (\\tau_{p, 0})^{q+1} +
        \\sum_{i=1}^{q+1} w_i t^i,

    where

    .. math::

        \\tau_p(t) = \\frac{\\Vert \\mathbf{A} + t \\mathbf{B} \\Vert_p}
        {\\Vert \\mathbf{B} \\Vert_p},

    and :math:`\\tau_{p, 0} = \\tau_p(0)` and :math:`w_{q+1} = 1`.
    To find the weight coefficient :math:`w_1`, the trace is computed at the
    given interpolant point :math:`ti`` argument.

    Since in this class, :math:`q = 1`, meaning that there is only one
    interpolant point :math:`t_1` with the function value  :math:`\\tau_{p, 1}
    = \\tau_p(t_1)`, the weight coefficient :math:`w_1` can be solved easily.
    In this case, the interpolation function becomes

    .. math::

        (\\tau_p(t))^2 \\approx  \\tau_{p, 0}^2 + t^2 +
        \\left( \\tau_{p, 1}^2 - \\tau_{p, 0}^2 -
        t_1^2 \\right) \\frac{t}{t_1}.

    The above interpolation is a quadratic function of :math:`t`. Hence, if
    :math:`p=2`, the above interpolation coincides with the exact function
    value for all range of :math:`t`. Because of this, it is recommended to
    use this interpolation method when :math:`p=2`.

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
    interpolating points :math:`t_i = 10^{-1}`.

    .. code-block:: python
        :emphasize-lines: 9, 13

        >>> # Generate two sample matrices (symmetric and positive-definite)
        >>> from imate.sample_matrices import correlation_matrix
        >>> A = correlation_matrix(size=20, scale=1e-1)
        >>> B = correlation_matrix(size=20, scale=2e-2)

        >>> # Initialize interpolator object
        >>> from imate import InterpolateSchatten
        >>> ti = 1e-1
        >>> f = InterpolateSchatten(A, B, p=2, kind='mbf', ti=ti)

        >>> # Interpolate at an inquiry point t = 0.4
        >>> t = 4e-1
        >>> f(t)
        1.7374809371539675

    Alternatively, call :meth:`imate.InterpolateSchatten.interpolate` to
    interpolate at points `t`:

    .. code-block:: python

        >>> # This is the same as f(t)
        >>> f.interpolate(t)
        1.7374809371539675

    To evaluate the exact value of the Schatten norm at point `t` without
    interpolation, call :meth:`imate.InterpolateSchatten.eval` function:

    .. code-block:: python

        >>> # This evaluates the function value at t exactly (no interpolation)
        >>> f.eval(t)
        1.7374809371539675

    It can be seen that the result of interpolation matches the exact function
    value. This is because we set :math:`p=2` and in this case,
    interpolation with `mbf` method yields identical results to the exact
    solution (see Notes in the above). However, this is not the case for
    :math:`p \\neq 2`.

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
        >>> f = InterpolateSchatten(A, B, p=2, options=options, kind='mbf')
        >>> f(t)
        1.7012047720355232

    You may get a different result than the above as the `slq` method is a
    randomized method.

    Also, in the above, the interpolation point ``ti`` was not specified, so
    the algorithm chooses the best interpolation point.

    **Interpolate on Range of Points:**

    Once the interpolation object ``f`` in the above example is
    instantiated, calling :meth:`imate.InterpolateSchatten.interpolate` on
    a list of inquiry points `t` has almost no computational cost. The next
    example inquires interpolation on `1000` points `t`:

    Interpolate an array of inquiry points ``t_array``:

    .. code-block:: python

        >>> # Create an interpolator object again
        >>> ti = 1e-1
        >>> f = InterpolateSchatten(A, B, kind='mbf', ti=ti)

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

    .. image:: ../_static/images/plots/interpolate_schatten_ext_eig.png
        :align: center
        :class: custom-dark

    Since in the above example, :math:`p=2`, the result of interpolation is the
    same as the exact function values, hence the error is zero for all
    :math:`t` as shown in the plot on the right side.
    """

    # ====
    # Init
    # ====

    def __init__(self, A, B=None, p=2, options={}, verbose=False, ti=[]):
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
            # Convert to an array of size one
            if isinstance(ti, list):
                ti = numpy.array(ti)
            elif isinstance(ti, Number):
                ti = numpy.array([ti])

            # Check number of interpolant points
            if ti.size != 1:
                raise TypeError("'ti' for the 'mbf' method should be a ' + \
                                'single number, or an array or a list of ' + \
                                'a single number.")

            self.t1 = ti[0]

        # Initialize interpolator
        self.tau1 = None
        self.initialize_interpolator()

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

        # Base class member data. These are only needed for plotting
        self.t_i = numpy.array([self.t1])
        self.q = self.t_i.size
        self.schatten_i = schatten_1
        self.tau_i = self.tau1

        # Scale interpolant points
        self.scale_t = self.find_scale_ti()

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
        the interpolation object is initialized.

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
