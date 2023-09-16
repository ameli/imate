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

from .interpolate_schatten import InterpolateSchatten
import numpy
from numbers import Number

try:
    from .._utilities.plot_utilities import *                # noqa: F401, F403
    from .._utilities.plot_utilities import load_plot_settings, matplotlib, \
        show_or_save_plot, plt
    plot_modules_exist = True
except ImportError:
    plot_modules_exist = False


# ==================
# Interpolate Logdet
# ==================

class InterpolateLogdet(InterpolateSchatten):
    """
    Interpolate the log-deterinant of an affine matrix function.

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

    options : dict, default={}
        At each interpolation point :math:`t_i`, the logdeterminant is computed
        using the function :func:`imate.logdet`. The ``options`` passes a
        dictionary of arguments to this functions.

    verbose : bool, default=False
        If `True`, it prints some information about the computation process.

    kind : {`'ext'`, `'eig'`, `'mbf'`, `'imbf'`, `'rbf'`, `'crf'`, `'spl'`, \
            `'rpf'`}, default: `'imbf'`
        The algorithm of interpolation, which are the same as the algorithms
        in :class:`imate.InterpolateSchatten`. See documentation for each
        specific algorithm:

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

    imate.InterpolateSchatten
    imate.InterpolateTrace
    imate.logdet

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

    Methods
    -------

    __call__
    eval
    interpolate
    get_scale
    lower_bound
    plot

    Notes
    -----

    **Interpolation of Affine Matrix Function:**

    This class interpolates the log-determinant of the one-parameter matrix
    function:

    .. math::

        t \\mapsto \\left( \\mathbf{A} + t \\mathbf{B} \\right)^p,

    where the matrices :math:`\\mathbf{A}` and :math:`\\mathbf{B}` are
    symmetric and positive semi-definite and
    :math:`t \\in [t_{\\inf}, \\infty)` is a real parameter where
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

    Interpolate the log-determinant of the affine matrix function
    :math:`(\\mathbf{A} + t \\mathbf{B})` using ``imbf`` algorithm and the
    interpolating points :math:`t_i = [10^{-2}, 10^{-1}, 1, 10]`.

    .. code-block:: python
        :emphasize-lines: 9, 13

        >>> # Generate two sample matrices (symmetric and positive-definite)
        >>> from imate.sample_matrices import correlation_matrix
        >>> A = correlation_matrix(size=20, scale=1e-1)
        >>> B = correlation_matrix(size=20, scale=2e-2)

        >>> # Initialize interpolator object
        >>> from imate import InterpolateLogdet
        >>> ti = [1e-2, 1e-1, 1, 1e1]
        >>> f = InterpolateLogdet(A, B, kind='imbf', ti=ti)

        >>> # Interpolate at an inquiry point t = 0.4
        >>> t = 4e-1
        >>> f(t)
        2.514109025242562

    Alternatively, call :meth:`imate.InterpolateLogdet.interpolate` to
    interpolate at points `t`:

    .. code-block:: python

        >>> # This is the same as f(t)
        >>> f.interpolate(t)
        2.514109025242562

    To evaluate the exact value of the log-determinant at point `t` without
    interpolation, call :meth:`imate.InterpolateLogdet.eval` function:

    .. code-block:: python

        >>> # This evaluates the function value at t exactly (no interpolation)
        >>> f.eval(t)
        2.498580941943615

    It can be seen that the relative error of interpolation compared to the
    exact solution in the above is :math:`0.62 \\%` using only four
    interpolation points :math:`t_i`, which is a remarkable result.

    .. warning::

        Calling :meth:`imate.InterpolateLogdet.eval` may take a longer time
        to compute as it computes the function exactly. Particularly, if `t` is
        a large array, it may take a very long time to return the exact values.

    **Arguments Specific to Algorithms:**

    In the above example, the ``imbf`` algorithm is used. See more arguments
    of this algorithm at :ref:`imate.InterpolateSchatten.imbf`. In the next
    example, we pass ``basis_func_type`` specific to this algorithm:

    .. code-block:: python
        :emphasize-lines: 3

        >>> # Passing kwatgs specific to imbf algorithm
        >>> f = InterpolateLogdet(A, B, kind='imbf', ti=ti,
        ...                       basis_func_type='ortho2')
        >>> f(t)
        2.514109025242562

    You may choose other algorithms using ``kind`` argument. For instance, the
    next example uses the Chebyshev rational functions with an additional
    argument ``func_type`` specific to this method. See more details at
    :ref:`imate.InterpolateSchatten.crf`.

    .. code-block:: python
        :emphasize-lines: 3

        >>> # Generate two sample matrices (symmetric and positive-definite)
        >>> f = InterpolateLogdet(A, B, kind='crf', ti=ti, func_type=1)
        >>> f(t)
        2.4961565270080617

    **Passing Options:**

    The above examples, the internal computation is passed to
    :func:`imate.logdet` function. You can pass arguments to the latter
    function using ``options`` argument. To do so, create a dictionary with the
    keys as the name of the argument. For instance, to use
    :ref:`imate.logdet.slq` method with ``min_num_samples=20`` and
    ``max_num_samples=100``, create the following dictionary:

    .. code-block:: python

        >>> # Specify arguments as a dictionary
        >>> options = {
        ...     'method': 'slq',
        ...     'min_num_samples': 20,
        ...     'max_num_samples': 100
        ... }

        >>> # Pass the options to the interpolator
        >>> f = InterpolateLogdet(A, B, options=options, kind='imbf', ti=ti)
        >>> f(t)
        2.5387905608340637

    You may get a different result than the above as the `slq` method is a
    randomized method.

    **Interpolate on Range of Points:**

    Once the interpolation object ``f`` in the above example is
    instantiated, calling :meth:`imate.InterpolateLogdet.interpolate` on
    a list of inquiry points `t` has almost no computational cost. The next
    example inquires interpolation on `1000` points `t`:

    Interpolate an array of inquiry points ``t_array``:

    .. code-block:: python

        >>> # Create an interpolator object again
        >>> ti = [1e-2, 1e-1, 1, 1e1]
        >>> f = InterpolateLogdet(A, B, ti=ti)

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
        >>> plt.ylim([0, 40])
        >>> plt.xlabel('$t$')
        >>> plt.ylabel('$\\mathrm{logdet} (\\mathbf{A} + t \\mathbf{B})$')
        >>> plt.title('Interpolation of Log-Determinant')
        >>> plt.show()

    .. image:: ../_static/images/plots/interpolate_logdet_1.png
        :align: center
        :class: custom-dark
        :width: 60%

    **Plotting Interpolation and Compare with Exact Solution:**

    A more convenient way to plot the interpolation result is to call
    :meth:`imate.InterpolateLogdet.plot` function.

    .. code-block:: python

        >>> f.plot(t_array)

    .. image:: ../_static/images/plots/interpolate_logdet_2.png
        :align: center
        :class: custom-dark
        :width: 60%

    In the above, :math:`f_p(t) = \\mathrm{logdet} (\\mathbf{A} +
    t \\mathbf{B})`. If you set ``normalize`` to `True`, it plots the
    normalized function

    .. math::

        g_p(t) = \\mathrm{logdet}(\\mathbf{A} + t \\mathbf{B}) -
        \\mathrm{logdet}(\\mathbf{B}).

    To compare with the true values (without interpolation), pass
    ``compare=True`` to the above function.

    .. warning::

        By setting ``compare`` to `True`, every point in the array `t` is
        evaluated both using interpolation and with the exact method (no
        interpolation). If the size of `t` is large, this may take a very
        long run time.

    .. code-block:: python

        >>> f.plot(t_array, normalize=True, compare=True)

    .. image:: ../_static/images/plots/interpolate_logdet_3.png
        :align: center
        :class: custom-dark
    """

    # ====
    # init
    # ====

    def __init__(self, A, B=None, options={}, verbose=False, kind='imbf',
                 ti=[], **kwargs):
        """
        Initializes the object depending on the method.
        """

        # In Schatten operator, p=0 corresponds to determinant**(1/n)
        super(InterpolateLogdet, self).__init__(
                A, B=B, p=0, ti=ti, kind=kind, verbose=verbose,
                options=options, **kwargs)

    # ==================
    # schatten to logdet
    # ==================

    def _schatten_to_logdet(self, schatten):
        """
        Converts Schatten anti-norm to logdet.
        """

        logdet_ = self.n * numpy.log(schatten)
        return logdet_

    # ==================
    # logdet to schatten
    # ==================

    def _logdet_to_schatten(self, logdet_):
        """
        Converts logdet to Schatten anti-norm.
        """

        schatten = numpy.exp(logdet_ / self.n)
        return schatten

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
        accessed by :meth:`imate.InterpolateLogdet.get_scale` function.

        .. code-block:: python
            :emphasize-lines: 11

            >>> # Generate two sample matrices (symmetric positive-definite)
            >>> from imate.sample_matrices import correlation_matrix
            >>> A = correlation_matrix(size=20, scale=1e-1)

            >>> # Initialize interpolator object
            >>> from imate import InterpolateLogdet
            >>> ti = [1e-2, 1e-1, 1, 1e1]
            >>> f = InterpolateLogdet(A, kind='crf', ti=ti, scale=None)

            >>> f.get_scale()
            0.8010742187499998
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

        This function calls :func:`InterpolateLogdet.interpolate` method.

        Parameters
        ----------

        t : float or array_like[float]
            An inquiry point (or list of points) to interpolate.

        Returns
        -------

        logdet : float or numpy.array
            Interpolated values. If the input `t` is a list or array, the
            output is an array of the same size of `t`.

        See Also
        --------

        imate.InterpolateLogdet.interpolate
        imate.InterpolateLogdet.eval

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12, 17

            >>> # Generate two sample matrices (symmetric positive-definite)
            >>> from imate.sample_matrices import correlation_matrix
            >>> A = correlation_matrix(size=20, scale=1e-1)

            >>> # Initialize interpolator object
            >>> from imate import InterpolateLogdet
            >>> ti = [1e-2, 1e-1, 1, 1e1]
            >>> f = InterpolateLogdet(A, ti=ti)

            >>> # Interpolate at an inquiry point t = 0.4
            >>> t = 4e-1
            >>> f(t)
            2.879736857573098

            >>> # Array of input points
            >>> t = [4e-1, 4, 4e+1]
            >>> f(t)
            array([ 2.87973686, 31.88562138, 74.1393325 ])
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

        imate.InterpolateLogdet.__call__
        imate.InterpolateLogdet.interpolate

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12, 17

            >>> # Generate two sample matrices (symmetric positive-definite)
            >>> from imate.sample_matrices import correlation_matrix
            >>> A = correlation_matrix(size=20, scale=1e-1)

            >>> # Initialize interpolator object
            >>> from imate import InterpolateLogdet
            >>> ti = [1e-2, 1e-1, 1, 1e1]
            >>> f = InterpolateLogdet(A, ti=ti)

            >>> # Exact function value at an inquiry point t = 0.4
            >>> t = 4e-1
            >>> f.eval(t)
            2.862148595502184

            >>> # Array of input points
            >>> t = [4e-1, 4, 4e+1]
            >>> f.eval(t)
            array([19.86369141,  4.12685164,  0.49096648])
        """

        schatten = super(InterpolateLogdet, self).eval(t)

        return self._schatten_to_logdet(schatten)

    # ===========================
    # compare with exact solution
    # ===========================

    def _compare_with_exact_solution(self, t, logdet_):
        """
        Computes the log-determinant with exact method (no interpolation), then
        compares it with the interpolated solution.

        Parameters
        ----------
            t : numpy.array
                Inquiry points

            Trace: float or numpy.array
                The interpolated computation of log-determinant.

        Returns
        -------
            exact : float or numpy.array
                Exact solution of log-determinant.

            relative_error : float or numpy.array
                Relative error of interpolated solution compared to the exact
                solution.
        """

        if self.kind.lower() == 'ext':

            # The Trace results are already exact. No need to recompute again.
            logdet_exact = logdet_
            logdet_relative_error = numpy.zeros(t.shape)

        else:

            # Compute exact solution
            logdet_exact = self.eval(t)
            logdet_relative_error = (logdet_ - logdet_exact) / (logdet_exact)

        return logdet_exact, logdet_relative_error

    # ===========
    # interpolate
    # ===========

    def interpolate(self, t):
        """
        Interpolate at the input point `t`.

        .. note::

            You may alternatively, call :func:`InterpolateLogdet.__call__`
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

        imate.InterpolateLogdet.__call__
        imate.InterpolateLogdet.eval

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 12, 17

            >>> # Generate two sample matrices (symmetric positive-definite)
            >>> from imate.sample_matrices import correlation_matrix
            >>> A = correlation_matrix(size=20, scale=1e-1)

            >>> # Initialize interpolator object
            >>> from imate import InterpolateLogdet
            >>> ti = [1e-2, 1e-1, 1, 1e1]
            >>> f = InterpolateLogdet(A, ti=ti)

            >>> # Interpolate at an inquiry point t = 0.4
            >>> t = 4e-1
            >>> f.interpolate(t)
            2.879736857573098

            >>> # Array of input points
            >>> t = [4e-1, 4, 4e+1]
            >>> f.interpolate(t)
            array([ 2.87973686, 31.88562138, 74.1393325 ])
        """

        schatten = super(InterpolateLogdet, self).interpolate(t)
        return self._schatten_to_logdet(schatten)

    # ===========
    # lower bound
    # ===========

    def lower_bound(self, t):
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

        imate.InterpolateSchatten.bound

        Notes
        -----

        A lower bound for :math:`\\mathrm{logdet}(\\mathbf{A} + t \\mathbf{B})`
        is obtained as follows. Define

        .. math::

                \\Vert \\mathbf{A} \\Vert_0 =
                \\left| \\mathrm{det}(\\mathbf{A}) \\right|^{\\frac{1}{n}}

        Also, let

        .. math::

            \\tau_0(t) = \\frac{
            \\Vert \\mathbf{A} + t \\mathbf{B} \\Vert_0}
            {\\Vert \\mathbf{B} \\Vert_0}

        and :math:`\\tau_{0, 0} = \\tau_0(0)`. A sharp bound of the function
        :math:`\\tau_0(y)` is (see [1]_, Section 3):

        .. math::

                \\tau_{0}(t) \\geq \\tau_{0, 0} + t, \\quad
                t \\in [0, \\infty).

        The above inequality originate from the `Brunn-Minkowski` determinant
        inequality.

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
            >>> from imate import InterpolateLogdet
            >>> ti = [1e-2, 1e-1, 1, 1e1]
            >>> f = InterpolateLogdet(A, B, ti=ti)

        Create an array `t` and evaluate upper bound on `t`. Also, interpolate
        the function :math:`f` on the array `t`.

        .. code-block:: python
            :emphasize-lines: 4

            >>> # Interpolate at an array of points
            >>> import numpy
            >>> t = numpy.logspace(-2, 1, 1000)
            >>> lb = f.lower_bound(t)
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
            >>> plt.semilogx(t, lb, '--', color='black', label='Lower Bound')
            >>> plt.xlim([t[0], t[-1]])
            >>> plt.ylim([-10, 50])
            >>> plt.xlabel('$t$')
            >>> plt.ylabel('$\\mathrm{logdet}(\\mathbf{A} + t \\mathbf{B})$')
            >>> plt.title('Interpolation of Log-Determinant')
            >>> plt.legend()
            >>> plt.show()

        .. image:: ../_static/images/plots/interpolate_logdet_lb.png
            :align: center
            :class: custom-dark
            :width: 70%
        """

        schatten_lb = super(InterpolateLogdet, self).bound(t)

        logdet_lb = self._schatten_to_logdet(schatten_lb)
        return logdet_lb

    # ====
    # plot
    # ====

    def plot(
            self,
            t,
            normalize=False,
            compare=False):
        """
        Plot the interpolation results.

        Parameters
        ----------

        t : numpy.array
            Inquiry points to be interpolated.

        normalize : bool, default: False
            If set to `False` the function
            :math:`f_p(t) = \\mathrm{logdet} (\\mathbf{A} + t \\mathbf{B})` is
            plotted. If set to `True`, the following normalized function is
            plotted:

            .. math::

                g_p(t) = \\mathrm{logdet}(\\mathbf{A} + t \\mathbf{B}) -
                \\mathrm{logdet}(\\mathbf{B}).

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
            >>> from imate import InterpolateLogdet
            >>> ti = [1e-2, 1e-1, 1, 1e1]
            >>> f = InterpolateLogdet(A, B, ti=ti)

        Define an array if inquiry point `t` and call
        :meth:`imate.InterpolateLogdet.plot` function to plot the
        interpolation of the function :math:`f(t)`:

        .. code-block:: python

            >>> import numpy
            >>> t_array = numpy.logspace(-2, 1, 1000)
            >>> f.plot(t_array)

        .. image:: ../_static/images/plots/interpolate_logdet_2.png
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

        .. image:: ../_static/images/plots/interpolate_logdet_3.png
            :align: center
            :class: custom-dark
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

        # Generate interpolation
        logdet_interpolated = self.interpolate(t)

        if compare:
            logdet_exact = self.eval(t)

        # Normalize logdet to tau
        if normalize:
            schatten_B = self.interpolator.schatten_B

            # EXT and EIG methods do not compute schatten_B by default.
            if schatten_B is None:
                schatten_B = self.interpolator._compute_schatten(
                        self.interpolator.B, self.interpolator.p)

            normal_factor = self._schatten_to_logdet(schatten_B)
        else:
            normal_factor = 0.0

        if self.interpolator.schatten_i is not None:
            logdet_i = self._schatten_to_logdet(self.interpolator.schatten_i)
            tau_i = logdet_i - normal_factor
        tau_interpolated = logdet_interpolated - normal_factor
        if compare:
            tau_exact = logdet_exact - normal_factor
            tau_absolute_error = tau_interpolated - tau_exact

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
                ax[0].semilogx(self.interpolator.t_i, tau_i, 'o',
                               color=exact_color, markersize=markersize,
                               label='Interpolant points', zorder=20)

        # Plot exact values
        if compare:
            ax[0].semilogx(t, tau_exact, color=exact_color, label='Exact')

        # Plot interpolated results
        ax[0].semilogx(t, tau_interpolated, color=interp_color,
                       label='Interpolated')

        ax[0].grid(axis='x')
        ax[0].set_xlim([t[0], t[-1]])
        ax[0].set_xlabel(r'$t$')

        if normalize:
            ax[0].set_ylabel(r'$g_0(t)$')
        else:
            ax[0].set_ylabel(r'$f_0(t)$')

        if normalize:
            ax0_title = r'Interpolation of $g_0(t)$'
        else:
            ax0_title = r'Interpolation of $f_0(t)$'
        if compare:
            ax0_title = r'(a) ' + ax0_title

        ax[0].set_title(ax0_title)
        ax[0].legend(fontsize='small')

        # Plot absolute error in percent
        if compare:
            if self.interpolator.q > 0:
                ax[1].semilogx(self.interpolator.t_i,
                               numpy.zeros(self.interpolator.q), 'o',
                               color=exact_color, markersize=markersize,
                               label='Interpolant points', zorder=20)
            ax[1].semilogx(t, tau_absolute_error, color=interp_color,
                           label='Interpolated')
            ax[1].grid(axis='x')
            ax[1].semilogx(ax[1].get_xlim(), [0, 0], color='#CCCCCC',
                           linewidth=0.75)
            ax[1].set_xlim([t[0], t[-1]])
            ax[1].set_xlabel('$t$')
            if normalize:
                ax[1].set_ylabel(r'$g_{\mathrm{approx}}(t) - ' +
                                 r'g_{\mathrm{exact}}(t)$')
            else:
                ax[1].set_ylabel(r'$f_{\mathrm{approx}}(t) - ' +
                                 r'f_{\mathrm{exact}}(t)$')
            ax1_title = r'(b) Absolute error of interpolation'
            ax[1].set_title(ax1_title)
            ax[1].legend(fontsize='small')

        plt.tight_layout()

        # Check if the graphical backend exists
        if matplotlib.get_backend() != 'agg':
            plt.show()
        else:
            # Save the plot as SVG file in the current directory
            show_or_save_plot(plt, 'interpolation',
                              transparent_background=True)
