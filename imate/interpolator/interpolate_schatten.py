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

import numpy
from numbers import Number

# Classes, Files
from ._exact_method import ExactMethod
from ._eigenvalues_method import EigenvaluesMethod
from ._monomial_basis_functions_method import MonomialBasisFunctionsMethod
from ._inverse_monomial_basis_functions_method import \
        InverseMonomialBasisFunctionsMethod
from ._radial_basis_functions_method import RadialBasisFunctionsMethod
from ._rational_polynomial_functions_method import \
        RationalPolynomialFunctionsMethod
from ._chebyshev_rational_functions_method import \
        ChebyshevRationalFunctionsMethod
from ._spline_method import SplineMethod

try:
    from .._utilities.plot_utilities import *                # noqa: F401, F403
    from .._utilities.plot_utilities import load_plot_settings, matplotlib, \
        show_or_save_plot, plt
    plot_modules_exist = True
except ImportError:
    plot_modules_exist = False


# ====================
# Interpolate Schatten
# ====================

class InterpolateSchatten(object):
    """
    Interpolate Schatten norm (or anti-norm) of an affine matrix function.

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

    kind : {`'ext'`, `'eig'`, `'mbf'`, `'imbf'`, `'rbf'`, `'crf'`, `'spl'`, \
            `'rpf'`}, default: `'imbf'`
        The algorithm of interpolation. See documentation for each specific
        algorithm:

        * :ref:`imate.InterpolateSchatten.ext`
        * :ref:`imate.InterpolateSchatten.eig`
        * :ref:`imate.InterpolateSchatten.mbf`
        * :ref:`imate.InterpolateSchatten.imbf`
        * :ref:`imate.InterpolateSchatten.rbf`
        * :ref:`imate.InterpolateSchatten.crf`
        * :ref:`imate.InterpolateSchatten.spl`
        * :ref:`imate.InterpolateSchatten.rpf`

    ti : float or array_like(float), default=None
        Interpolation points, which can be a single point, a list, or an array
        of points. The interpolator honors the exact function values at the
        interpolant points. If an empty list is given, `i.e.`, ``[]``,
        depending on the algorithm, a default list of interpolant points is
        used. Also, the size of the array of `ti` depends on the algorithm as
        follows. If ``kind`` is:

        * ``ext`` or ``eig``, the no ``ti`` is required.
        * ``mbf``, then a single point ``ti`` may be specified.
        * ``imbf``, ``spl``, or ``rbf``, then a list of ``ti`` with arbitrary
          size may be specified.
        * ``crf`` or ``rpf``, then a list of ``ti`` points with even size may
          be specified.

    kwargs : \\*\\*kwargs
        Additional arguments to pass to each specific algorithm. See
        documentation for each ``kind`` in the above.

    See Also
    --------

    imate.InterpolateTrace
    imate.InterpolateLogdet
    imate.schatten

    Attributes
    ----------

    kind : str
        Method of interpolation

    verbose : bool
        Verbosity of the computation process

    n : int
        Size of the matrix

    q : int
        Number of interpolant points

    p : float
        Order of Schatten :math:`p`-norm

    Methods
    -------

    __call__
    eval
    interpolate
    get_scale
    bound
    upper_bound
    plot

    Notes
    -----

    **Schatten Norm:**

    In this class, the Schatten :math:`p`-norm of the matrix
    :math:`\\mathbf{A}` is defined by

    .. math::
        :label: schatten-eq-3

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
        factor :math:`\\frac{1}{n}` in :math:numref:`schatten-eq-3`. However,
        this factor is justified by the continuity granted by

        .. math::
            :label: schatten-continuous-3

            \\lim_{p \\to 0} \\Vert \\mathbf{A} \\Vert_p =
            \\Vert \\mathbf{A} \\Vert_0.

        See [1]_ (Section 2) and the examples in :func:`imate.schatten` for
        details.

    **Interpolation of Affine Matrix Function:**

    This class interpolates the Schatten norm (or anti-norm) of the
    one-parameter matrix function:

    .. math::

        \\tau_p: t \\mapsto \\| \\mathbf{A} + t \\mathbf{B} \\|_p,

    where the matrices :math:`\\mathbf{A}` and :math:`\\mathbf{B}` are
    symmetric and positive semi-definite (positive-definite if :math:`p < 0`)
    and :math:`t \\in [t_{\\inf}, \\infty)` is a real parameter where
    :math:`t_{\\inf}` is the minimum :math:`t` such that
    :math:`\\mathbf{A} + t_{\\inf} \\mathbf{B}` remains positive-definite.

    The interpolator is initialized by providing :math:`q` interpolant points
    :math:`t_i`, :math:`i = 1, \\dots, q`, which are often given in a
    logarithmically spaced interval :math:`t_i \\in [t_1, t_p]`. The
    interpolator can interpolate the above function at arbitrary inquiry points
    :math:`t \\in [t_1, t_p]` using various methods.

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
        >>> f = InterpolateSchatten(A, B, p=2, kind='imbf', ti=ti)

        >>> # Interpolate at an inquiry point t = 0.4
        >>> t = 4e-1
        >>> f(t)
        1.7328745175033962

    Alternatively, call :meth:`imate.InterpolateSchatten.interpolate` to
    interpolate at points `t`:

    .. code-block:: python

        >>> # This is the same as f(t)
        >>> f.interpolate(t)
        1.7328745175033962

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

    **Arguments Specific to Algorithms:**

    In the above example, the ``imbf`` algorithm is used. See more arguments
    of this algorithm at :ref:`imate.InterpolateSchatten.imbf`. In the next
    example, we pass ``basis_func_type`` specific to this algorithm:

    .. code-block:: python
        :emphasize-lines: 3

        >>> # Passing kwatgs specific to imbf algorithm
        >>> f = InterpolateSchatten(A, B, p=2, kind='imbf', ti=ti,
        ...                         basis_func_type='ortho2')
        >>> f(t)
        1.7328745175033962

    You may choose other algorithms using ``kind`` argument. For instance, the
    next example uses the Chebyshev rational functions with an additional
    argument ``func_type`` specific to this method. See more details at
    :ref:`imate.InterpolateSchatten.crf`.

    .. code-block:: python
        :emphasize-lines: 3

        >>> # Generate two sample matrices (symmetric and positive-definite)
        >>> f = InterpolateSchatten(A, B, p=2, kind='crf', ti=ti,
        ...                         func_type=1)
        >>> f(t)
        1.737489512386539

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
        >>> f = InterpolateSchatten(A, B, p=2, options=options, kind='imbf',
        ...                         ti=ti)
        >>> f(t)
        1.7291144488066212

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
        >>> ti = [1e-2, 1e-1, 1, 1e1]
        >>> f = InterpolateSchatten(A, B, ti=ti)

        >>> # Interpolate at an array of points
        >>> import numpy
        >>> t_array = numpy.logspace(-2, 1, 1000)
        >>> norm_array = f.interpolate(t_array)

    One may plot the above interpolated results as follows:

    .. code-block:: python

        >>> import matplotlib.pyplot as plt
        >>> import seaborn as sns

        >>> # Plot settings (optional)
        >>> sns.set(font_scale=1.15)
        >>> sns.set_style("white")
        >>> sns.set_style("ticks")

        >>> plt.semilogx(t_array, norm_array, color='black')
        >>> plt.xlim([t_array[0], t_array[-1]])
        >>> plt.ylim([0, 10])
        >>> plt.xlabel('$t$')
        >>> plt.ylabel('$\\Vert \\mathbf{A} + t \\mathbf{B} \\Vert_{2}$')
        >>> plt.title('Interpolation of Schatten 2-Norm')
        >>> plt.show()

    .. image:: ../_static/images/plots/interpolate_schatten_1.png
        :align: center
        :class: custom-dark
        :width: 50%

    **Plotting Interpolation and Compare with Exact Solution:**

    A more convenient way to plot the interpolation result is to call
    :meth:`imate.InterpolateSchatten.plot` function.

    .. code-block:: python

        >>> f.plot(t_array)

    .. image:: ../_static/images/plots/interpolate_schatten_2.png
        :align: center
        :class: custom-dark
        :width: 60%

    In the above, :math:`\\tau_p = \\Vert \\mathbf{A} + t \\mathbf{B}
    \\Vert_p`. If you set ``normalize`` to `True`, it plots the normalized
    function

    .. math::

        \\frac{\\Vert \\mathbf{A} + t \\mathbf{B} \\Vert_{p}}{
        \\Vert \\mathbf{B} \\Vert_p}.

    To compare with the true values (without interpolation), pass
    ``compare=True`` to the above function.

    .. warning::

        By setting ``compare`` to `True`, every point in the array `t` is
        evaluated both using interpolation and with the exact method (no
        interpolation). If the size of `t` is large, this may take a very
        long run time.

    .. code-block:: python

        >>> f.plot(t_array, normalize=True, compare=True)

    .. image:: ../_static/images/plots/interpolate_schatten_3.png
        :align: center
        :class: custom-dark
    """

    # ====
    # init
    # ====

    def __init__(self, A, B=None, p=2, options={}, verbose=False, kind='imbf',
                 ti=[], **kwargs):
        """
        Initializes the object depending on the method.
        """

        # Attributes
        self.kind = kind
        self.verbose = verbose
        self.n = A.shape[0]
        if ti is not None and not numpy.isscalar(ti):
            self.q = len(ti)
        else:
            self.q = 0
        self.p = p

        # Define an interpolation object depending on the given method
        if kind.lower() == 'ext':
            # Exact computation, not interpolation
            self.interpolator = ExactMethod(
                    A, B, p=self.p, options=options, verbose=verbose, **kwargs)

        elif kind.lower() == 'eig':
            # Eigenvalues kind
            self.interpolator = EigenvaluesMethod(
                    A, B, p=self.p, options=options, verbose=verbose, **kwargs)

        elif kind.lower() == 'mbf':
            # Monomial Basis Functions kind
            self.interpolator = MonomialBasisFunctionsMethod(
                    A, B, p=self.p, options=options, verbose=verbose, ti=ti,
                    **kwargs)

        elif kind.lower() == 'imbf':
            # Inverse Monomial Basis Functions kind
            self.interpolator = InverseMonomialBasisFunctionsMethod(
                    A, B, p=self.p, options=options, verbose=verbose, ti=ti,
                    **kwargs)

        elif kind.lower() == 'rbf':
            # Radial Basis Functions kind
            self.interpolator = RadialBasisFunctionsMethod(
                    A, B, p=self.p, options=options, verbose=verbose, ti=ti,
                    **kwargs)

        elif kind.lower() == 'crf':
            # Chebyshev Rational Functions
            self.interpolator = ChebyshevRationalFunctionsMethod(
                    A, B, p=self.p, options=options, verbose=verbose, ti=ti,
                    **kwargs)

        elif kind.lower() == 'spl':
            # Spline
            self.interpolator = SplineMethod(
                    A, B, p=self.p, options=options, verbose=verbose, ti=ti,
                    **kwargs)

        elif kind.lower() == 'rpf':
            # Rational Polynomial Functions kind
            self.interpolator = RationalPolynomialFunctionsMethod(
                    A, B, p=self.p, options=options, verbose=verbose, ti=ti,
                    **kwargs)

        else:
            raise ValueError("'kind' is invalid. Select one of 'ext', " +
                             "'eig', 'mbf', 'imbf', 'rbf', 'crf', 'spl', or " +
                             "'rpf'.")

    # =========
    # get scale
    # =========

    def get_scale(self):
        """
        Returns the scale parameter of the interpolator.

        This function can be called if ``kind=crf`` or ``kind=spl``.

        Returns
        -------

        scale : float
            Scale parameter of the interpolator.

        Raises
        ------

        NotImplementedError
            If ``kind`` is not ``crf``.

        Examples
        --------

        In the following scale, since the input argument ``scale`` is set to
        `None`, it i automatically generated by optimization. Its value can be
        accessed by :meth:`imate.InterpolateSchatten.get_scale` function.

        .. code-block:: python
            :emphasize-lines: 11

            >>> # Generate two sample matrices (symmetric positive-definite)
            >>> from imate.sample_matrices import correlation_matrix
            >>> A = correlation_matrix(size=20, scale=1e-1)

            >>> # Initialize interpolator object
            >>> from imate import InterpolateSchatten
            >>> ti = [1e-2, 1e-1, 1, 1e1]
            >>> f = InterpolateSchatten(A, p=2, kind='crf', ti=ti, scale=None)

            >>> f.get_scale()
            1.4101562500000009
        """

        if self.kind.lower() != 'crf':
            raise NotImplementedError('This function can be called only if' +
                                      '"kind" is set to "crf"')

        scale = self.interpolator.scale

        if not isinstance(scale, Number):
            scale = scale[0]

        return scale

    # ========
    # __call__
    # ========

    def __call__(self, t):
        """
        Interpolate at the input point `t`.

        This function calls :func:`InterpolateSchatten.interpolate` method.

        Parameters
        ----------

        t : float or array_like[float]
            An inquiry point (or list of points) to interpolate.

        Returns
        -------

        norm : float or numpy.array
            Interpolated values. If the input `t` is a list or array, the
            output is an array of the same size of `t`.

        See Also
        --------

        imate.InterpolateSchatten.interpolate
        imate.InterpolateSchatten.eval

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12, 17

            >>> # Generate two sample matrices (symmetric positive-definite)
            >>> from imate.sample_matrices import correlation_matrix
            >>> A = correlation_matrix(size=20, scale=1e-1)

            >>> # Initialize interpolator object
            >>> from imate import InterpolateSchatten
            >>> ti = [1e-2, 1e-1, 1, 1e1]
            >>> f = InterpolateSchatten(A, ti=ti)

            >>> # Interpolate at an inquiry point t = 0.4
            >>> t = 4e-1
            >>> f(t)
            1.7116166617477382

            >>> # Array of input points
            >>> t = [4e-1, 4, 4e+1]
            >>> f(t)
            array([ 1.71161666,  5.09395567, 41.32038322])
        """

        return self.interpolate(t)

    # ====
    # eval
    # ====

    def eval(self, t):
        """
        Evaluate the exact value of the function at the input point `t` without
        interpolation.

        .. warning::

            If `t` is an array of large size, this may take a very long run
            time as all input points are evaluated without.

        Parameters
        ----------

        t : float or array_like[float]
            An inquiry point (or list of points) to interpolate.

        Returns
        -------

        norm : float or numpy.array
            Exact values of the function. If the input `t` is a list or array,
            the output is an array of the same size of `t`.

        See Also
        --------

        imate.InterpolateSchatten.__call__
        imate.InterpolateSchatten.interpolate

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12, 17

            >>> # Generate two sample matrices (symmetric positive-definite)
            >>> from imate.sample_matrices import correlation_matrix
            >>> A = correlation_matrix(size=20, scale=1e-1)

            >>> # Initialize interpolator object
            >>> from imate import InterpolateSchatten
            >>> ti = [1e-2, 1e-1, 1, 1e1]
            >>> f = InterpolateSchatten(A, ti=ti)

            >>> # Exact function value at an inquiry point t = 0.4
            >>> t = 4e-1
            >>> f.eval(t)
            1.7175340160001527

            >>> # Array of input points
            >>> t = [4e-1, 4, 4e+1]
            >>> f.eval(t)
            array([ 1.71753402,  5.0980313 , 41.01207046])
        """

        if isinstance(t, Number):
            # Single number
            T = self.interpolator.eval(t)

        else:
            # An array of points
            T = numpy.empty((len(t), ), dtype=float)
            for i in range(len(t)):
                T[i] = self.interpolator.eval(t[i])

        return T

    # ===========================
    # compare with exact solution
    # ===========================

    def _compare_with_exact_solution(self, t, schatten):
        """
        Compute the trace with exact method (no interpolation), then compares
        it with the interpolated solution.

        Parameters
        ----------

        t : numpy.array
            Inquiry points

        schatten : float or numpy.array
            The interpolated computation of trace.

        Returns
        -------

        exact : float or numpy.array
            Exact solution of trace.

        relative_error : float or numpy.array
            Relative error of interpolated solution compared to the exact
            solution.
        """

        if self.kind.lower() == 'ext':

            # The Trace results are already exact. No need to recompute again.
            schatten_exact = schatten
            schatten_relative_error = numpy.zeros(t.shape)

        else:

            # Compute exact solution
            schatten_exact = self.eval(t)
            schatten_relative_error = 1.0 - (schatten / schatten_exact)

        return schatten_exact, schatten_relative_error

    # ===========
    # interpolate
    # ===========

    def interpolate(self, t):
        """
        Interpolate at the input point `t`.

        .. note::

            You may alternatively, call :func:`InterpolateSchatten.__call__`
            method.

        Parameters
        ----------

        t : float or array_like[float]
            An inquiry point (or list of points) to interpolate.

        Returns
        -------

        norm : float or numpy.array
            Interpolated values. If the input `t` is a list or array, the
            output is an array of the same size of `t`.

        See Also
        --------

        imate.InterpolateSchatten.__call__
        imate.InterpolateSchatten.eval

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12, 17

            >>> # Generate two sample matrices (symmetric positive-definite)
            >>> from imate.sample_matrices import correlation_matrix
            >>> A = correlation_matrix(size=20, scale=1e-1)

            >>> # Initialize interpolator object
            >>> from imate import InterpolateSchatten
            >>> ti = [1e-2, 1e-1, 1, 1e1]
            >>> f = InterpolateSchatten(A, ti=ti)

            >>> # Interpolate at an inquiry point t = 0.4
            >>> t = 4e-1
            >>> f.interpolate(t)
            1.7116166617477382

            >>> # Array of input points
            >>> t = [4e-1, 4, 4e+1]
            >>> f.interpolate(t)
            array([ 1.71161666,  5.09395567, 41.32038322])
        """

        if isinstance(t, Number):
            # Single number
            schatten = self.interpolator.interpolate(t)

        else:
            # An array of points
            schatten = numpy.empty((len(t), ), dtype=float)
            for i in range(len(t)):
                schatten[i] = self.interpolator.interpolate(t[i])

        return schatten

    # =====
    # bound
    # =====

    def bound(self, t):
        """
        Bound of the interpolation function.

        If :math:`p < 1`, this function is a lower bound, and if :math:`p > 1`,
        this function is an upper bound of the interpolation function.

        Parameters
        ----------

        t : float or numpy.array
            An inquiry point or an array of inquiry points.

        Returns
        -------

        bound : float or numpy.array
            Bound function evaluated at `t`. If `t` is an array, the output is
            also an array of the size of `t`.

        See Also
        --------

        imate.InterpolateSchatten.upper_bound

        Notes
        -----

        Define

        .. math::

                \\tau_{p}(t) = \\frac{\\Vert \\mathbf{A} + t \\mathbf{B}
                \\Vert_p}{\\Vert \\mathbf{B} \\Vert_p}

        and :math:`\\tau_{p,0} = \\tau_{p}(0)`. A sharp bound of the function
        :math:`\\tau_{p}` is (see [1]_, Section 3):

        .. math::

                \\tau_{p}(t) \\leq \\tau_{p, 0} + t, \\quad p \\geq 1, \\quad
                t \\in [0, \\infty),

        .. math::

                \\tau_{p}(t) \\geq \\tau_{p, 0} + t, \\quad p < 1, \\quad
                t \\in [0, \\infty),

        .. math::

                \\tau_{p}(t) \\geq \\tau_{p, 0} + t, \\quad p \\geq 1, \\quad
                t \\in (t_{\\inf}, o],

        .. math::

                \\tau_{p}(t) \\leq \\tau_{p, 0} + t, \\quad p < 1, \\quad
                t \\in (t_{\\inf}, o],

        References
        ----------

        .. [1] Ameli, S., and Shadden. S. C. (2022). Interpolating
               Log-Determinant and Trace of the Powers of Matrix
               :math:`\\mathbf{A} + t \\mathbf{B}`.
               *Statistics and Computing* 32, 108.
               `https://doi.org/10.1007/s11222-022-10173-4
               <https://doi.org/10.1007/s11222-022-10173-4>`_.

        Examples
        --------

        Create an interpolator object :math:`f` using four interpolant points
        :math:`t_i`:

        .. code-block:: python

            >>> # Generate sample matrices (symmetric positive-definite)
            >>> from imate.sample_matrices import correlation_matrix
            >>> A = correlation_matrix(size=20, scale=1e-1)
            >>> B = correlation_matrix(size=20, scale=2e-2)

            >>> # Initialize interpolator object
            >>> from imate import InterpolateSchatten
            >>> ti = [1e-2, 1e-1, 1, 1e1]
            >>> f = InterpolateSchatten(A, B, p=2, ti=ti)

        Create an array `t` and evaluate upper bound on `t`. Also, interpolate
        the function :math:`f` on the array `t`.

        .. code-block:: python
            :emphasize-lines: 4

            >>> # Interpolate at an array of points
            >>> import numpy
            >>> t = numpy.logspace(-2, 1, 1000)
            >>> bound = f.bound(t)
            >>> interp = f.interpolate(t)

        Plot the results:

        .. code-block:: python

            >>> import matplotlib.pyplot as plt
            >>> import seaborn as sns

            >>> # Plot settings (optional)
            >>> sns.set(font_scale=1.15)
            >>> sns.set_style("white")
            >>> sns.set_style("ticks")

            >>> plt.semilogx(t, interp, color='black', label='Interpolation')
            >>> plt.semilogx(t, bound, '--', color='black',
            ...              label='Bound')
            >>> plt.xlim([t[0], t[-1]])
            >>> plt.ylim([0, 10])
            >>> plt.xlabel('$t$')
            >>> plt.ylabel('$\\Vert \\mathbf{A} + t \\mathbf{B} \\Vert_{2}$')
            >>> plt.title('Interpolation of Schatten Norm')
            >>> plt.legend()
            >>> plt.show()

        .. image:: ../_static/images/plots/interpolate_schatten_bound.png
            :align: center
            :class: custom-dark
            :width: 70%
        """

        if isinstance(t, Number):
            # Single number
            schatten_bound = self.interpolator.bound(t)

        else:
            # An array of points
            schatten_bound = numpy.empty((len(t), ), dtype=float)
            for i in range(len(t)):
                schatten_bound[i] = self.interpolator.bound(t[i])

        return schatten_bound

    # ===========
    # upper bound
    # ===========

    def upper_bound(self, t):
        """
        Upper bound of the interpolation function.

        .. note::

            This function only applies to :math:`p=-1`.

        Parameters
        ----------

        t : float or numpy.array
            An inquiry point or an array of inquiry points.

        Returns
        -------

        ub : float or numpy.array
            Upper bound. If `t` is an array, the output is also an array of the
            size of `t`.

        Raises
        ------

        ValueError
            If :math:`p \\neq -1`.

        See Also
        --------

        imate.InterpolateSchatten.bound

        Notes
        -----

        A upper bound of the function
        :math:`\\Vert \\mathbf{A} + t \\mathbf{B} \\Vert_{-1}` is (see [1]_)

        .. math::

            \\Vert \\mathbf{A} + t \\mathbf{B} \\Vert_{-1} \\leq
            \\frac{\\mathrm{trace}(\\mathbf{A}) +
            t \\mathrm{trace}(\\mathbf{B})}{n}.

        Note that the above bound is not sharp as :math:`t \\to 0`.

        References
        ----------

        .. [1] Ameli, S., and Shadden. S. C. (2022). Interpolating
               Log-Determinant and Trace of the Powers of Matrix
               :math:`\\mathbf{A} + t \\mathbf{B}`.
               *Statistics and Computing* 32, 108.
               `https://doi.org/10.1007/s11222-022-10173-4
               <https://doi.org/10.1007/s11222-022-10173-4>`_.

        Examples
        --------

        Create an interpolator object :math:`f` using four interpolant points
        :math:`t_i`:

        .. code-block:: python

            >>> # Generate sample matrices (symmetric positive-definite)
            >>> from imate.sample_matrices import correlation_matrix
            >>> A = correlation_matrix(size=20, scale=1e-1)
            >>> B = correlation_matrix(size=20, scale=2e-2)

            >>> # Initialize interpolator object
            >>> from imate import InterpolateSchatten
            >>> ti = [1e-2, 1e-1, 1, 1e1]
            >>> f = InterpolateSchatten(A, B, p=-1, ti=ti)

        Create an array `t` and evaluate upper bound on `t`. Also, interpolate
        the function :math:`f` on the array `t`.

        .. code-block:: python
            :emphasize-lines: 4

            >>> # Interpolate at an array of points
            >>> import numpy
            >>> t = numpy.logspace(-2, 1, 1000)
            >>> ub = f.upper_bound(t)
            >>> interp = f.interpolate(t)

        Plot the results:

        .. code-block:: python

            >>> import matplotlib.pyplot as plt
            >>> import seaborn as sns

            >>> # Plot settings (optional)
            >>> sns.set(font_scale=1.15)
            >>> sns.set_style("white")
            >>> sns.set_style("ticks")

            >>> plt.semilogx(t, interp, color='black', label='Interpolation')
            >>> plt.semilogx(t, ub, '--', color='black', label='Upper bound')
            >>> plt.xlim([t[0], t[-1]])
            >>> plt.ylim([0, 10])
            >>> plt.xlabel('$t$')
            >>> plt.ylabel('$\\Vert \\mathbf{A} + t \\mathbf{B} \\Vert_{-1}$')
            >>> plt.title('Interpolation of Schatten Anti-Norm')
            >>> plt.legend()
            >>> plt.show()

        .. image:: ../_static/images/plots/interpolate_schatten_ub.png
            :align: center
            :class: custom-dark
            :width: 70%
        """

        if isinstance(t, Number):
            # Single number
            schatten_ub = self.interpolator.upper_bound(t)

        else:
            # An array of points
            schatten_ub = numpy.empty((len(t), ), dtype=float)
            for i in range(len(t)):
                schatten_ub[i] = self.interpolator.upper_bound(t[i])

        return schatten_ub

    # ====
    # plot
    # ====

    def plot(
            self,
            t,
            normalize=True,
            compare=False):
        """
        Plot the interpolation results.

        Parameters
        ----------

        t : numpy.array
            Inquiry points to be interpolated.

        normalize : bool, default: False
            If set to `False` the function :math:`\\tau_p = \\Vert \\mathbf{A}
            + t \\mathbf{B} \\Vert_p` is plotted. If set to `True`, it plots
            the normalized function

            .. math::

                \\tau_p(t) = \\frac{\\Vert \\mathbf{A} + t \\mathbf{B}
                \\Vert_{p}}{\\Vert \\mathbf{B} \\Vert_p}.

        compare : bool, default=False
            If `True`, it computes the exact function values (without
            interpolation), then compares it with the interpolated solution to
            estimate the relative error of interpolation.

            .. note::

                When this option is enabled, the exact solution will be
                computed for all inquiry points, which can take a very long
                time.

        Raises
        ------

        ImportError
            If `matplotlib` and `seaborn` are not installed.

        ValueError
            If ``t`` is not an array of size greater than one.

        Notes
        -----

        **Plotted Function:**

        * When ``kind`` is either of ``ext``, ``eig``, ``mbf``, ``imbf``,
          ``rpf``, or ``rbf``, the function :math:`\\tau_p(t)` is plotted,
          which is defined as follows. If ``normalize`` is `False`, then

          .. math::

              \\tau_p(t) = \\Vert \\mathbf{A} + t \\mathbf{B} \\Vert_p.

          If ``normalize`` is `True`, then

          .. math::

              \\frac{\\Vert \\mathbf{A} + t \\mathbf{B} \\Vert_{p}}{
              \\Vert \\mathbf{B} \\Vert_p}.

        * On the other hand, if ``kind=crf`` or ``kind=spl``, the function
          :math:`y_p(x)` is plotted, which is defined as follows:

          .. math::

              x = \\frac{t - \\alpha}{t + \\alpha},

          where :math:`\\alpha` is the ``scale`` parameter in
          :ref:`imate.InterpolateSchatten.crf` for ``kind=crf``. If
          ``kind=spl``, then :math:`\\alpha=1`. Also

          .. math::

              y_p = \\frac{\\tau_p(t)}{\\tau_p(0) + t} - 1.

        **Graphical Backend:**

        * If no graphical backend exists (such as running the code on a
          remote server or manually disabling the X11 backend), the plot
          will not be shown, rather, it will ve saved as an ``svg`` file in
          the current directory.
        * If the executable ``latex`` is on the path, the plot is rendered
          using :math:`\\rm\\LaTeX`, which then, it takes a bit
          longer to produce the plot.
        * If :math:`\\rm\\LaTeX` is not installed, it uses any available
          San-Serif font to render the plot.

        To manually disable interactive plot display, and save the plot as
        ``SVG`` instead, add the following in the very beginning of your code
        before importing ``imate``:

        .. code-block:: python

            >>> import os
            >>> os.environ['IMATE_NO_DISPLAY'] = 'True'

        Examples
        --------

        Create an interpolator object :math:`f` using four interpolant points
        :math:`t_i`:

        .. code-block:: python

            >>> # Generate sample matrices (symmetric positive-definite)
            >>> from imate.sample_matrices import correlation_matrix
            >>> A = correlation_matrix(size=20, scale=1e-1)
            >>> B = correlation_matrix(size=20, scale=2e-2)

            >>> # Initialize interpolator object
            >>> from imate import InterpolateSchatten
            >>> ti = [1e-2, 1e-1, 1, 1e1]
            >>> f = InterpolateSchatten(A, B, ti=ti)

        Define an array if inquiry point `t` and call
        :meth:`imate.InterpolateSchatten.plot` function to plot the
        interpolation of the function :math:`f(t)`:

        .. code-block:: python

            >>> import numpy
            >>> t_array = numpy.logspace(-2, 1, 1000)
            >>> f.plot(t_array)

        .. image:: ../_static/images/plots/interpolate_schatten_2.png
            :align: center
            :class: custom-dark
            :width: 60%

        To compare with the true values (without interpolation), pass
        ``compare=True`` to the above function.

        .. warning::

            By setting ``compare`` to `True`, every point in the array `t` is
            evaluated both using interpolation and with the exact method (no
            interpolation). If the size of `t` is large, this may take a very
            long run time.

        .. code-block:: python

            >>> f.plot(t_array, normalize=True, compare=True)

        .. image:: ../_static/images/plots/interpolate_schatten_3.png
            :align: center
            :class: custom-dark

        **Plotting for Chebyshev and Spline Methods:**

        As mentioned in the Notes in the above, if ``kind=crf`` or
        ``kind=spl``, the transformed function :math:`y_p(x)` is plotted as
        shown in the example below:

        .. code-block:: python

            >>> # Recreate interpolating object, but using crf method
            >>> f = InterpolateSchatten(A, B, kind='crf', ti=ti)
            >>> t_array = numpy.logspace(-2, 1, 1000)
            >>> f.plot(t_array)

        .. image:: ../_static/images/plots/interpolate_schatten_4.png
            :align: center
            :class: custom-dark
        """

        if self.kind.lower() in ['crf', 'spl']:
            # Plots tau where abscissa is the finite domain [-1, 1]
            self._plot_finite(t, normalize=normalize, compare=compare)
        else:
            # Plots tau where abscissa is the semi-infinite domain [0, inf)
            self._plot_semi_infinite(t, normalize=normalize, compare=compare)

    # ==================
    # plot semi infinite
    # ==================

    def _plot_semi_infinite(
            self,
            t,
            normalize=True,
            compare=False):
        """
        Plots the interpolation results, together with the comparison with the
        exact solution and the relative error of the interpolation.

        To plot, set ``Plot=True`` argument in
        :mod:`imate.InterpolateSchatten`.

        Parameters
        ----------
        t : numpy.array
            Inquiry points to be interpolated
        """

        if not plot_modules_exist:
            raise ImportError('Cannot import modules for plotting. Either ' +
                              'install "matplotlib" and "seaborn" packages, ' +
                              'or set "plot=False".')

        # Load plot settings
        try:
            load_plot_settings()
        except ImportError:
            raise ImportError('Cannot import modules for plotting. Either ' +
                              'install "matplotlib" and "seaborn" packages, ' +
                              'or set "plot=False".')

        # Check t should be an array
        if numpy.isscalar(t) or (t.size == 1):
            raise ValueError("Argument 't' should be an " +
                             "array of length greater than one to be able " +
                             " to plot results.")

        # Generate interpolation
        schatten_interpolated = self.interpolate(t)

        if compare:
            schatten_exact = self.eval(t)

        # Normalize schatten to tau
        if normalize:
            schatten_B = self.interpolator.schatten_B

            # EXT and EIG methods do not compute schatten_B by default.
            if schatten_B is None:
                schatten_B = self.interpolator._compute_schatten(
                        self.interpolator.B, self.interpolator.p)

            normal_factor = schatten_B
        else:
            normal_factor = 1.0

        if self.interpolator.schatten_i is not None:
            tau_i = self.interpolator.schatten_i / normal_factor
        tau_interpolated = schatten_interpolated / normal_factor

        if compare:
            tau_exact = schatten_exact / normal_factor
            tau_relative_error = 1.0 - (tau_interpolated / tau_exact)

        # Plot results
        if compare:
            # Two subplots
            fig, ax = plt.subplots(ncols=2, figsize=(9, 4))
        else:
            # One subplot
            fig, ax = plt.subplots(figsize=(5, 4))
            ax = [ax]

        # Plot settings
        markersize = 4
        exact_color = 'firebrick'
        interp_color = 'black'

        # Plot interpolant points with their exact values
        if self.interpolator.q > 0:
            if self.interpolator.schatten_i is not None:
                ax[0].loglog(self.interpolator.t_i, tau_i, 'o',
                             color=exact_color, markersize=markersize,
                             label='Interpolant points', zorder=20)

        # Plot exact values
        if compare:
            ax[0].loglog(t, tau_exact, color=exact_color,
                         label='Exact')

        # Plot interpolated results
        ax[0].loglog(t, tau_interpolated, color=interp_color,
                     label='Interpolated')

        if compare:
            tau_min = numpy.min([tau_exact, tau_interpolated])
            tau_max = numpy.max([tau_exact, tau_interpolated])
        else:
            tau_min = numpy.min([tau_interpolated])
            tau_max = numpy.max([tau_interpolated])
        tau_min_snap = 10**(numpy.floor(numpy.log10(tau_min)))
        tau_max_snap = 10**(numpy.round(numpy.log10(tau_max)))

        ax[0].grid(axis='x')
        ax[0].set_xlim([t[0], t[-1]])
        ax[0].set_ylim([tau_min_snap, tau_max_snap])
        ax[0].set_xlabel(r'$t$')

        if normalize:
            ax[0].set_ylabel(r'$\tau_p(t)$')
        else:
            ax[0].set_ylabel(r'$f_p(t)$')

        if normalize:
            ax0_title = r'Interpolation of $\tau_p(t)$, $p = %g$' % self.p
        else:
            ax0_title = r'Interpolation of $f_p(t)$, $p = %g$' % self.p
        if compare:
            ax0_title = r'(a) ' + ax0_title

        ax[0].set_title(ax0_title)
        ax[0].legend(fontsize='small')

        # Plot relative error in percent
        if compare:
            if self.interpolator.q > 0:
                ax[1].semilogx(self.interpolator.t_i,
                               numpy.zeros(self.interpolator.q), 'o',
                               color=exact_color, markersize=markersize,
                               label='Interpolant points', zorder=20)
            ax[1].semilogx(t, 100.0*tau_relative_error, color=interp_color,
                           label='Interpolated')
            ax[1].grid(axis='x')
            ax[1].semilogx(ax[1].get_xlim(), [0, 0], color='#CCCCCC',
                           linewidth=0.75)
            ax[1].set_xlim([t[0], t[-1]])
            ax[1].set_xlabel('$t$')
            if normalize:
                ax[1].set_ylabel(r'$1-\tau_{\mathrm{approx}}(t) / ' +
                                 r'\tau_{\mathrm{exact}}(t)$')
            else:
                ax[1].set_ylabel(r'$1-f_{\mathrm{approx}}(t) / ' +
                                 r'f_{\mathrm{exact}}(t)$')
            ax1_title = r'(b) Relative error of interpolation, $p=%g$' % self.p
            ax[1].set_title(ax1_title)
            tau_range = numpy.max(numpy.abs(100.0 * tau_relative_error))
            if tau_range != 0.0:
                decimals = int(numpy.ceil(-numpy.log10(tau_range))) + 1
            else:
                decimals = 2
            ax[1].yaxis.set_major_formatter(
                    matplotlib.ticker.PercentFormatter(decimals=decimals))
            ax[1].legend(fontsize='small')

        plt.tight_layout()

        # Check if the graphical backend exists
        if matplotlib.get_backend() != 'agg':
            plt.show()
        else:
            # Save the plot as SVG file in the current directory
            show_or_save_plot(plt, 'interpolation',
                              transparent_background=True)

    # ===========
    # plot finite
    # ===========

    def _plot_finite(
            self,
            t,
            normalize=False,
            compare=False):
        """
        Plots the interpolation results, together with the comparison with the
        exact solution and the relative error of the interpolation.

        To plot, set ``Plot=True`` argument in
        :mod:`imate.InterpolateSchatten`.

        Parameters
        ----------
        t : numpy.array
            Inquiry points to be interpolated

        schatten_interpolated : numpy.array
            Interpolation of the trace at inquiry

        schatten_exact : numpy.array, default=None
            Exact solutions of the trace at inquiry points. If this variable is
            not None, it will be plotted together with the interpolated
            results.

        schatten_relative_error : numpy.array, default=None
            Relative errors of the interpolation with respect to the exact
            solution. If not None, the relative errors will be plotted on a
            second axis.
        """

        if not plot_modules_exist:
            raise ImportError('Cannot import modules for plotting. Either ' +
                              'install "matplotlib" and "seaborn" packages, ' +
                              'or set "plot=False".')

        # Load plot settings
        try:
            load_plot_settings()
        except ImportError:
            raise ImportError('Cannot import modules for plotting. Either ' +
                              'install "matplotlib" and "seaborn" packages, ' +
                              'or set "plot=False".')

        # Check t should be an array
        if numpy.isscalar(t) or (t.size == 1):
            raise ValueError("Argument 't' should be an array of length " +
                             "greater than one to be able to plot results.")

        # If no data is provided, generate interpolation
        schatten_interpolated = self.interpolate(t)

        if compare:
            schatten_exact = self.eval(t)

        # Normalize schatten to tau
        if normalize:
            schatten_B = self.interpolator.schatten_B

            # EXT and EIG methods do not compute schatten_B by default.
            if schatten_B is None:
                schatten_B = self.interpolator._compute_schatten(
                        self.interpolator.B, self.interpolator.p)

            normal_factor = schatten_B
        else:
            normal_factor = 1.0

        if self.interpolator.schatten_i is not None:
            tau_i = self.interpolator.schatten_i / normal_factor
        tau_interpolated = schatten_interpolated / normal_factor

        if compare:
            tau_exact = schatten_exact / normal_factor
            tau_relative_error = 1.0 - (tau_interpolated / tau_exact)

        tau_0 = self.interpolator.tau0
        t_i = self.interpolator.t_i

        if self.kind.lower() == 'crf':
            scale = self.interpolator.scale
        else:
            scale = 1.0

        t_ = t / scale
        t_i_ = t_i / scale
        x = (t_-1.0) / (t_+1.0)
        x_i = (t_i_-1.0) / (t_i_+1.0)

        if self.interpolator.func_type == 1:
            if self.interpolator.schatten_i is not None:
                y_i = tau_i / (tau_0 + t_i) - 1.0
            y = tau_interpolated / (tau_0 + t) - 1.0
            if compare:
                y_ex = tau_exact / (tau_0 + t) - 1.0
        elif self.interpolator.func_type == 2:
            if self.interpolator.schatten_i is not None:
                y_i = (tau_i - tau_0) / t_i - 1.0
            y = (tau_interpolated - tau_0) / t - 1.0
            if compare:
                y_ex = (tau_exact - tau_0) / t - 1.0
        else:
            raise ValueError('"func_type" should be either 1 or 2.')

        # Plot results
        if compare:
            # Two subplots
            fig, ax = plt.subplots(ncols=2, figsize=(9, 4))
        else:
            # One subplot
            fig, ax = plt.subplots(figsize=(5, 4))
            ax = [ax]

        # Plot settings
        markersize = 4
        exact_color = 'firebrick'
        interp_color = 'black'

        # Plot interpolant points with their exact values
        if self.interpolator.q > 0:
            if self.interpolator.schatten_i is not None:
                ax[0].plot(x_i, y_i, 'o', color=exact_color,
                           markersize=markersize, label='Interpolant points',
                           zorder=20)

        # Plot exact values
        if compare:
            ax[0].plot(x, y_ex, color=exact_color,
                       label='Exact')

        # Plot interpolated results
        ax[0].plot(x, y, color=interp_color,
                   label='Interpolated')

        ax[0].grid(axis='x')
        ax[0].set_xlim([-1, 1])
        if numpy.all(y >= 0.0):
            ax[0].set_ylim(bottom=0, top=None)
        elif numpy.all(y <= 0.0):
            ax[0].set_ylim(bottom=None, top=0)
        if self.kind.lower() == 'spl':
            ax[0].set_xlabel(r'$(t-1) / (t+1)$')
        elif self.kind.lower() == 'crf':
            ax[0].set_xlabel(r'$(t-\alpha) / (t+\alpha)$')
        else:
            raise ValueError('"method" should be "SPL" or "CRF".')

        if normalize:
            if self.interpolator.func_type == 1:
                ax[0].set_ylabel(r'$\tau_p(t) / (\tau_{p, 0} + t) - 1$')
            elif self.interpolator.func_type == 2:
                ax[0].set_ylabel(r'$(\tau_p(t)-\tau_{p, 0}) / t - 1$')
        else:
            if self.interpolator.func_type == 1:
                ax[0].set_ylabel(r'$f_p(t) / (f_{p, 0} + t) - 1$')
            elif self.interpolator.func_type == 2:
                ax[0].set_ylabel(r'$(f_p(t)-f_{p, 0}) / t - 1$')

        if normalize:
            ax0_title = r'Interpolation of $\tau_p(t)$, $p = %g$' % self.p
        else:
            ax0_title = r'Interpolation of $f_p(t)$, $p = %g$' % self.p
        ax0_title = r'(a) ' + ax0_title

        ax[0].set_title(ax0_title)
        ax[0].legend(fontsize='small')

        if compare:
            # Plot relative error in percent
            if self.interpolator.q > 0:
                ax[1].semilogx(self.interpolator.t_i,
                               numpy.zeros(self.interpolator.q), 'o',
                               color=exact_color, markersize=markersize,
                               label='Interpolant points', zorder=20)
            ax[1].semilogx(t, 100.0*tau_relative_error,
                           color=interp_color, label='Interpolated')
            ax[1].grid(axis='x')
            ax[1].semilogx(ax[1].get_xlim(), [0, 0], color='#CCCCCC',
                           linewidth=0.75)
            ax[1].set_xlim([t[0], t[-1]])
            ax[1].set_xlabel('$t$')
            if normalize:
                ax[1].set_ylabel(r'$1-\tau_{\mathrm{approx}}(t) / ' +
                                 r'\tau_{\mathrm{exact}}(t)$')
            else:
                ax[1].set_ylabel(r'$1-f_{\mathrm{approx}}(t) / ' +
                                 r'f_{\mathrm{exact}}(t)$')
            ax1_title = r'(b) Relative error of interpolation, $p=%g$' % self.p
            ax[1].set_title(ax1_title)
            tau_range = numpy.max(numpy.abs(100.0 * tau_relative_error))
            if tau_range != 0.0:
                decimals = int(numpy.ceil(-numpy.log10(tau_range))) + 1
            else:
                decimals = 2
            ax[1].yaxis.set_major_formatter(
                    matplotlib.ticker.PercentFormatter(decimals=decimals))
            ax[1].legend(fontsize='small')

        plt.tight_layout()

        # Check if the graphical backend exists
        if matplotlib.get_backend() != 'agg':
            plt.show()
        else:
            # Save the plot as SVG file in the current directory
            show_or_save_plot(plt, 'interpolation',
                              transparent_background=True)
