# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

# Python
import time
import numpy
import scipy.sparse
from scipy.sparse import isspmatrix
import multiprocessing
from ..__version__ import __version__
from .._linear_algebra import linear_solver
from ._convergence_tools import check_convergence, average_estimates
from .._trace_estimator.trace_estimator_plot_utilities import plot_convergence
from ._hutchinson_method_utilities import check_arguments, print_summary
from .._linear_algebra.matrix_utilities import get_data_type_name, get_nnz, \
        get_density

# Cython
from .._c_basic_algebra cimport cVectorOperations
from .._linear_algebra cimport generate_random_column_vectors


# =================
# hutchinson method
# =================

def hutchinson_method(
        A,
        gram=False,
        p=1,
        return_info=False,
        B=None,
        C=None,
        assume_matrix='gen',
        min_num_samples=10,
        max_num_samples=50,
        error_atol=None,
        error_rtol=1e-2,
        confidence_level=0.95,
        outlier_significance_level=0.001,
        solver_tol=1e-6,
        orthogonalize=True,
        num_threads=0,
        verbose=False,
        plot=False):
    """
    Log-determinant of non-singular matrix or linear operator using stochastic
    Lanczos quadrature method.

    Given the symmetric matrix or linear operator :math:`\\mathbf{A}` and the
    real exponent :math:`p`, the following is computed:

    .. math::

        \\mathrm{logdet} \\left(\\mathbf{A}^p \\right) = p \\log_e \\vert
        \\det (\\mathbf{A}) \\vert.

    If ``gram`` is `True`, then :math:`\\mathbf{A}` in the above is replaced by
    the Gramian matrix :math:`\\mathbf{A}^{\\intercal} \\mathbf{A}`, and the
    following is instead computed:

    .. math::

        \\mathrm{logdet} \\left((\\mathbf{A}^{\\intercal}\\mathbf{A})^p
        \\right) = 2p \\log_e \\vert \\det (\\mathbf{A}) \\vert.

    If :math:`\\mathbf{A} = \\mathbf{A}(t)` is a linear operator of the class
    :class:`imate.AffineMatrixFunction` with the parameter :math:`t`, then for
    an input  tuple :math:`t = (t_1, \\dots, t_q)`, an array output of the
    size :math:`q` is returned, namely:

    .. math::

        \\mathrm{logdet} \\left((\\mathbf{A}(t_i))^p \\right),
        \\quad i=1, \\dots, q.

    Parameters
    ----------

    A : numpy.ndarray, scipy.sparse, :class:`imate.Matrix`, or \
            :class:`imate.AffineMatrixFunction`
        A non-singular sparse or dense matrix or linear operator. If ``gram``
        is `False`, then `A` should be symmetric.

        .. warning::

            The symmetry of `A` will not be checked by this function. If
            ``gram`` is `False`, make sure `A` is symmetric.

    gram : bool, default=False
        If `True`, the log-determinant of the Gramian matrix,
        :math:`(\\mathbf{A}^{\\intercal}\\mathbf{A})^p`, is computed. The
        Gramian matrix itself is not directly computed. If `False`, the
        log-determinant of :math:`\\mathbf{A}^p` is computed.

    p : float, default=1.0
        The exponent :math:`p` in :math:`\\mathbf{A}^p`.

    return_info : bool, default=False
        If `True`, this function also returns a dictionary containing
        information about the inner computation, such as process time,
        algorithm settings, etc.

    parameters : array_like [`float`], default=one
        This argument is relevant if `A` is a type of
        :class:`AffineMatrixFunction`. By this argument, multiple inquiries,
        :math:`(t_1, \\dots, t_q)`, can be passed to the parameter :math:`t` of
        the linear operator :math:`\\mathbf{A}(t)`. The output of this function
        becomes an array of the size :math:`q` corresponding to each of the
        input matrices :math:`\\mathbf{A}(t_i)`.

    min_num_samples : int, default=10
        The minimum number of Monte-Carlo samples. If the convergence criterion
        is reached before finishing the minimum number of iterations, the
        iterations are forced to continue till the minimum number of iterations
        is finished. This value should be smaller than
        ``maximum_num_samples``.

    max_num_samples : int, default=50
        The maximum number of Monte-Carlo samples. If the convergence criterion
        is not reached by the maximum number of iterations, the iterations are
        forced to stop. This value should be larger than
        ``minimum_num_samples``.

    error_atol : float, default=None
        Tolerance of the absolute error of convergence of the output. Once the
        error of convergence reaches ``error_atol + error_rtol * output``, the
        iteration is terminated. If the convergence criterion is not met by the
        tolerance, then the iterations continue till reaching
        ``max_num_samples`` iterations. If `None`, the termination criterion
        does not depend on this parameter.

    error_rtol : float, default=None
        Tolerance of the relative error of convergence of the output. Once the
        error of convergence reaches ``error_atol + error_rtol * output``, the
        iteration is terminated. If the convergence criterion is not met by the
        tolerance, then the iterations continue till reaching
        ``max_num_samples`` iterations. If `None`, the termination criterion
        does not depend on this parameter.

    confidence_level : float, default=0.95
        Confidence level of error, which is a number between `0` and `1`. The
        error of convergence of the population of samples is defined by their
        standard deviation times the Z-score, which depends on the confidence
        level. See notes below for details.

    outlier_significance_level : float, default=0.001
        One minus the confidence level of the uncertainty of the outliers of
        the output samples. This is a number between `0` and `1`.

    lanczos_degree : int, default=20
        The number of Lanczos recursive iterations. The larger Lanczos degree
        leads to better estimation. The computational cost quadratically
        increases with the Lanczos degree.

    lanczos_tol : float, default=None
        The tolerance to stop the Lanczos recursive iterations before
        the end of iterations reached. If the tolerance is not met, all the
        iterations (total of ``lanczos_degree`` iterations) continue till the
        end. If set to `None` (default value), the machine' epsilon precision
        is used. The machine's epsilon precision is as follows:

        * For 32-bit, machine precision is
          :math:`2^{-23} = 1.1920929 \\times 10^{-7}`.
        * For 64-bit, machine precision is
          :math:`2^{-52} = 2.220446049250313 \\times 10^{-16}`,
        * For 128-bit, machine precision is
          :math:`2^{-63} = -1.084202172485504434 \\times 10^{-19}`.

    orthogonalize : int, default=0
        Indicates whether to re-orthogonalize the eigenvectors during Lanczos
        recursive iterations.

        * If set to `0`, no orthogonalization is performed.
        * If set to a negative integer or an integer larger than
          ``lanczos_degree``, a newly computed eigenvector is orthogonalized
          against all the previous eigenvectors (also known as
          `full reorthogonalization`).
        * If set to a positive integer, say `q`, but less than
          ``lanczos_degree``, the newly computed eigenvector is
          orthogonalized against a window of last `q` previous eigenvectors
          (known as `partial reorthogonalization`).

    num_threads : int, default=0
        Number of processor threads to employ for parallel computation on CPU.
        If set to `0` or a number larger than the available number of threads,
        all threads of the processor are used. The parallelization is performed
        over the Monte-Carlo iterations.

    num_gpu_devices : int default=0
        Number of GPU devices (if available) to use for parallel multi-GPU
        processing. If set to `0`, the maximum number of available GPU devices
        is used. This parameter is relevant if ``gpu`` is `True`.

    gpu : bool, default=False
        If `True`, the computations are performed on GPU devices where the
        number of devices can be set by ``num_gpu_devices``. If no GPU device
        is found, it raises an error.

        .. note::
            When performing `repetitive` computation on the same matrix on GPU,
            it is recommended to input `A` as an instance of
            :class:`imate.Matrix` class instead of `numpy` or `scipy` matrices.
            See examples below for clarification.

    verbose : bool, default=False
        Prints extra information about the computations.

    plot : bool, default=False
        Plots convergence of samples. For this, the packages `matplotlib` and
        `seaborn` should be installed. If no display is available (such as
        running this code on remote machines), the plots are saved as an `SVG`
        file in the current directory.

    Returns
    -------

    logdet : float or numpy.array
        Log-determinant of `A`. If `A` is of type
        :class:`imate.AffineMatrixFunction` with an array of ``parameters``,
        then the output is an array.

    info : dict
        (Only if ``return_info`` is `True`) A dictionary of information with
        the following.

        * ``matrix``:
            * ``data_type``: `str`, {`float32`, `float64`, `float128`}. Type of
              the matrix data.
            * ``gram``: `bool`, whether the matrix `A` or its Gramian is
              considered.
            * ``exponent``: `float`, the exponent `p` in :math:`\\mathbf{A}^p`.
            * ``size``: (int) The size of matrix `A`.
            * ``sparse``: `bool`, whether the matrix `A` is sparse or dense.
            * ``nnz``: `int`, if `A` is sparse, the number of non-zero elements
              of `A`.
            * ``density``: `float`, if `A` is sparse, the density of `A`, which
              is the `nnz` divided by size squared.
            * ``num_inquiries``: `int`, the size of inquiries of each parameter
              of the linear operator `A`. If `A` is a matrix, this is always
              `1`. If `A` is a type of :class:`AffineMatrixFunction`, this
              value is the number of :math:`t_i` parameters.
            * ``num_operator_parameters``: `int`, number of parameters of the
              operator `A`. If `A` a type of :class:`AffineMatrixFunction`,
              then this value is `1` corresponding to one parameter :math:`t`
              in the affine function `t \\mapsto \\mathbf{A} + t \\mathbf{B}`.
            * ``parameters``: `list` [`float`], the parameters of the linear
              operator `A`.

        * ``convergence``:
            * ``all_converged``: `bool`, whether the Monte-Carlo sampling
              converged for all requested parameters :math:`t_i`. If all
              entries of the array for ``converged`` is `True``, then this
              value is also ``True``.
            * ``converged``: `array` [`bool`], whether the Monte-Carlo sampling
              converged for each of the requested parameters :math:`t_i`.
              Convergence is defined based on a termination criterion, such
              as absolute or relative error. If the iterations terminated due
              to reaching the maximum number of samples, this value is `False`.
            * ``min_num_samples``: `int`, the minimum number of Monte-Carlo
              iterations.
            * ``max_num_samples``: `int`, the maximum number of Monte-Carlo
              iterations.
            * ``num_outliers``: `int`, number of outliers found during search
              for outliers among the array of output.
            * ``num_samples_used``: `int`, number of Monte-Carlo samples used
              to produce the output. This is the total number of iterations
              minus the number of outliers.
            * ``samples``: `array` [`float`], an array of the size
              `max_num_samples`. The first few entries (`num_samples_used`) of
              this array are the output results of the Monte-Carlo sampling.
              The average of these samples is the final result. The rest of
              this array is `nan`.
            * ``samples_mean``: `float`, mean of the `samples` array, excluding
              the `nan` values.
            * ``samples_processed_order``: `array` [`int`], in parallel
              processing, samples are processed in non-sequential order. This
              array, which has the same size as `samples`, keeps track of the
              order in which each sample is processed.

        * ``error``:
            * ``absolute_error``: `float`, the absolute error of the
              convergence of samples.
            * ``confidence_level``: `float`, the confidence level used to
              calculate the error from standard deviation of samples.
            * ``error_atol``: `float`, the tolerance of absolute error of the
              convergence of samples.
            * ``error_rtol``: `float`, the tolerance of relative error of the
              convergence of samples.
            * ``outlier_significance_level``: `float`, the significance level
              used to determine the outliers in samples.
            * ``relative_error``: `float`, the relative error of the
              convergence of samples.

        * ``device``:
            * ``num_cpu_threads``: `int`, number of CPU threads used in shared
              memory parallel processing.
            * ``num_gpu_devices``: `int`, number of GPU devices used in the
              multi-GPU (GPU farm) computation.
            * ``num_gpu_multiprocessors``: `int`, number of GPU
              multi-processors.
            * ``num_gpu_threads_per_multiprocessor``: `int`, number of GPU
              threads on each GPU multi-processor.

        * ``time``:
            * ``tot_wall_time``: `float`, total elapsed time of computation.
            * ``alg_wall_time``: `float`, elapsed time of computation during
              only the algorithm execution.
            * ``cpu_proc_time``: `float`, the CPU processing time of
              computation.

        * ``solver``:
            * ``version``: `str`, version of imate.
            * ``method``: 'slq'

    Raises
    ------

    ImportError
        If the package has not been compiled with GPU support, but ``gpu`` is
        set to `True`. To resolve the issue, set ``gpu`` to `False` to be able
        to use the existing installation. Alternatively,  export the
        environment variable ``USE_CUDA=1`` and recompile the source code of
        the package.

    See Also
    --------

    imate.trace
    imate.traceinv
    imate.schatten

    Notes
    -----

    This method uses stochastic Lanczos quadrature (SLQ), which is a randomized
    algorithm. It can be used on very large matrices (:math:`n > 2^{12}`). The
    solution is an approximation.

    **Input Matrix:**

    The input `A` can be either of:

    * A matrix, such as `numpy.ndarray`, or `scipy.sparse`.
    * A linear operator representing a matrix using :class:`imate.Matrix`.
    * A linear operator representing a one-parameter family of an affine matrix
      function :math:`t \\mapsto \\mathbf{A} + t\\mathbf{B}` using
      :class:`imate.AffineMatrixFunction`.

    **Output:**

    The output is a scalar. However, if `A` is the linear operator
    :math:`\\mathbf{A}(t) = \\mathbf{A} + t \\mathbf{B}` where :math:`t` is
    given as the tuple :math:`t = (t_1, \\dots, t_q)`, then the output of this
    function is an array of size :math:`q` corresponding to the
    log-determinant of each :math:`\\mathbf{A}(t_i)`.

    .. note::

        When `A` represents
        :math:`\\mathbf{A}(t) = \\mathbf{A} + t \\mathbf{I}`, where
        :math:`\\mathbf{I}` is the identity matrix, and :math:`t` is given by
        a tuple :math:`t = (t_1, \\dots, t_q)`, the computational cost of an
        array output of size `q` is the same as computing for a single
        :math:`t_i`. Namely, the log-determinant of only
        :math:`\\mathbf{A}(t_1)` is computed, and the log-determinant of the
        rest of :math:`i=2, \\dots, q` are obtained from the result of
        :math:`t_1` immediately.

    **Algorithm:**

    If ``gram`` is `False`, the Lanczos tri-diagonalization method is
    used. This method requires only matrix-vector multiplication. If
    ``gram`` is `True`, the Golub-Kahn bi-diagonalization method is used. This
    method requires both matrix-vector multiplication and transposed-matrix
    vector multiplications.

    **Convergence criterion:**

    Let :math:`n_{\\min}` and :math:`n_{\\max}` be the minimum and maximum
    number of iterations respectively defined by ``min_num_samples`` and
    ``max_num_samples``. The iterations terminate at
    :math:`n_{\\min} \\leq i \\leq n_{\\max}` where :math:`i` is the
    iteration counter. The iterations stop earlier at :math:`i < n_{\\max}` if
    the convergence error of the mean of the samples is satisfied, as follows.

    Suppose :math:`s(j)` and :math:`\\sigma(i)` are respectively the mean and
    standard deviation of samples after :math:`j` iterations. The error of
    convergence, :math:`e(j)`, is defined by

    .. math::

        e(j) = \\frac{\\sigma(j)}{\\sqrt{j}} Z

    where :math:`Z` is the Z-score defined by

    .. math::

        Z = \\sqrt{2} \\mathrm{erf}^{-1}(\\phi).

    In the above, :math:`\\phi` is the confidence level and set by
    ``confidence_level`` argument, and :math:`\\mathrm{erf}^{-1}` is the
    inverse error function. A confidence level of 95%, for instance, means that
    the Z-score is 1.96, which means the confidence interval is
    :math:`\\pm 1.96 \\sigma`.

    The termination criterion is

    .. math::

        e(j) < \\epsilon_a + s(j) \\epsilon_r,

    where :math:`\\epsilon_{a}` and :math:`\\epsilon_r` are the absolute and
    relative error tolerances respectively, and they are set by ``error_atol``
    and ``error_rtol``.

    **Convergence for the case of multiple parameters:**

    When `A` is a type of :class:`imate.AffineMatrixFunction` representing the
    affine matrix function :math:`\\mathbf{A}(t) = \\mathbf{A} + t \\mathbf{B}`
    and if multiple parameters :math:`t_i`, :math:`i=1,\\dots, q` are passed to
    this function through ``parameters`` argument, the convergence criterion
    has to be satisfied for each of :math:`\\mathbf{A}(t_i)`. Specifically, the
    iterations are terminated as follows:

    * If :math:`\\mathbf{B}` is the identity matrix, iterations for all
      :math:`\\mathbf{A}(t_i)` continue till the convergence criterion for
      *all* :math:`t_i` are satisfied. That is, even if :math:`t=t_i` is
      converged but :math:`t=t_j` has not converged yet, the iterations for
      :math:`t=t_i` will continue.
      :
    * If :math:`\\mathbf{B}` is not the identity matrix, the iterations for
      each of :math:`t_i` are independent. That is, if :math:`t=t_i` converges,
      the iterations for that parameter will stop regardless of the convergence
      status of other parameters.

    **Plotting:**

    If ``plot`` is set to `True`, it plots the convergence of samples and their
    relative error.

    * If no graphical backend exists (such as running the code on a remote
      server or manually disabling the X11 backend), the plot will not be
      shown, rather, it will be saved as an ``svg`` file in the current
      directory.
    * If the executable ``latex`` is available on ``PATH``, the plot is
      rendered using :math:`\\rm\\LaTeX` and it may take slightly longer to
      produce the plot.
    * If :math:`\\rm\\LaTeX` is not installed, it uses any available San-Serif
      font to render the plot.

    To manually disable interactive plot display and save the plot as
    ``svg`` instead, add the following at the very beginning of your code
    before importing :mod:`imate`:

    .. code-block:: python

        >>> import os
        >>> os.environ['IMATE_NO_DISPLAY'] = 'True'

    References
    ----------

    * `Ubaru, S., Chen, J., and Saad, Y. (2017)
      <https://www-users.cs.umn.edu/~saad/PDF/ys-2016-04.pdf>`_,
      Fast Estimation of :math:`\\mathrm{tr}(F(A))` Via Stochastic Lanczos
      Quadrature, SIAM J. Matrix Anal. Appl., 38(4), 1075-1099.

    Examples
    --------

    **Large matrix:**

    Compute log-determinant of a very large sparse matrix using at least `100`
    samples:

    .. code-block:: python
        :emphasize-lines: 9, 10


        >>> # Import packages
        >>> from imate import toeplitz, logdet

        >>> # Generate a matrix of size one million
        >>> A = toeplitz(2, 1, size=1000000, gram=True)

        >>> # Approximate log-determinant using stochastic Lanczos quadrature
        >>> # with at least 100 Monte-Carlo sampling
        >>> ld, info = logdet(A, method='slq', min_num_samples=100,
        ...                   max_num_samples=200, return_info=True)
        >>> print(ld)
        1386320.4734751645

        >>> # Find the time it took to compute the above
        >>> print(info['time'])
        {
            'tot_wall_time': 16.598053652094677,
            'alg_wall_time': 16.57977867126465,
            'cpu_proc_time': 113.03275911399999
        }

    Compare the result of the above approximation with the exact solution of
    the log-determinant using the analytic relation for Toeplitz matrix. See
    :func:`imate.sample_matrices.toeplitz_logdet` for details.

    .. code-block:: python

        >>> from imate.sample_matrices import toeplitz_logdet
        >>> toeplitz_logdet(2, 1, size=1000000, gram=True)
        1386294.3611198906

    It can be seen that the error of approximation is :math:`0.0018 \\%`. This
    accuracy is remarkable considering that the computation on such a large
    matrix took only 16 seconds. Computing the log-determinant of such a
    large matrix using any of the exact methods (such as ``cholesky`` or
    ``eigenvalue``) is infeasible.

    **Output information:**

    Print information about the inner computation:

    .. code-block:: python

        >>> ld, info = logdet(A, method='slq', return_info=True)
        >>> print(ld)
        138.6294361119891

        >>> # Print dictionary neatly using pprint
        >>> from pprint import pprint
        >>> pprint(info)
        {
            'matrix': {
                'data_type': b'float64',
                'density': 2.999998e-06,
                'exponent': 1.0,
                'gram': False,
                'nnz': 2999998,
                'num_inquiries': 1,
                'num_operator_parameters': 0,
                'parameters': None,
                'size': 1000000,
                'sparse': True
            },
            'convergence': {
                'all_converged': True,
                'converged': True,
                'max_num_samples': 50,
                'min_num_samples': 10,
                'num_outliers': 0,
                'num_samples_used': 10,
                'samples': array([1386085.91975074, ..., nan]),
                 'samples_mean': 1385604.1663613867,
                'samples_processed_order': array([ 6, ..., 0]),
            },
            'error': {
                'absolute_error': 467.54178690512845,
                'confidence_level': 0.95,
                'error_atol': 0.0,
                'error_rtol': 0.01,
                'outlier_significance_level': 0.001,
                'relative_error': 0.0003374281041120848
            },
            'solver': {
                'lanczos_degree': 20,
                'lanczos_tol': 2.220446049250313e-16,
                'method': 'slq',
                'orthogonalize': 0,
                'version': '0.13.0'
            },
            'device': {
                'num_cpu_threads': 8,
                'num_gpu_devices': 0,
                'num_gpu_multiprocessors': 0,
                'num_gpu_threads_per_multiprocessor': 0
            },
            'time': {
                'alg_wall_time': 2.6905629634857178,
                'cpu_proc_time': 18.138382528999998,
                'tot_wall_time': 2.69458799099084
            }
        }

    **Verbose output:**

    By setting ``verbose`` to `True`, useful info about the process is
    printed.

    .. literalinclude:: ../_static/imate.logdet.slq-verbose-1.txt
        :language: python

    **Plotting:**

    By setting ``plot`` to `True`, plots of samples during Monte-Carlo
    iterations and the convergence of their mean are generated.

    .. code-block:: python

        >>> A = toeplitz(2, 1, size=1000000, gram=True)
        >>> logdet(A, method='slq', min_num_samples=20, max_num_samples=80,
        ...        error_rtol=2e-4, confidence_level=0.95,
        ...        outlier_significance_level=0.001, plot=True)

    .. image:: ../_static/images/plots/slq_convergence_1.png
        :align: center
        :class: custom-dark

    In the left plot, the samples are shown in circles and the cumulative mean
    of the samples is shown by a solid black curve. The shaded area corresponds
    to the 95% confidence interval :math:`\\pm 1.96 \\sigma`, which is set by
    ``confidence_level=0.95``. The samples outside the interval of 99.9% are
    considered outliers, which is set by the significance level
    ``outlier_significance_level=0.001``.

    In the right plot, the darker shaded area in the interval :math:`[0, 20]`
    shows the minimum number of samples and is set by ``min_num_samples=20``.
    The iterations do not stop till the minimum number of iterations is passed.
    We can observe that sampling is terminated after 55 iterations where the
    relative error of samples reaches 0.02% since we set ``error_rtol=2e-4``.
    The lighter shaded area in the interval :math:`[56, 80]` corresponds to the
    iterations that were not performed to reach the specified maximum
    iterations by ``max_num_samples=80``.

    **Matrix operator:**

    Use an object of :class:`imate.Matrix` class as an alternative method to
    pass the matrix `A` to the `logdet` function.

    .. code-block:: python
        :emphasize-lines: 8

        >>> # Import matrix operator
        >>> from imate import toeplitz, logdet, Matrix

        >>> # Generate a sample matrix (a toeplitz matrix)
        >>> A = toeplitz(2, 1, size=100, gram=True)

        >>> # Create a matrix operator object from matrix A
        >>> Aop = Matrix(A)

        >>> # Compute log-determinant of Aop
        >>> logdet(Aop, method='slq')
        141.52929878934194

    An advantage of passing `Aop` (instead of `A`) to the `logdet` function
    will be clear when using GPU.

    **Computation on GPU:**

    The argument ``gpu=True`` performs the computations on GPU. The following
    example uses the object `Aop` created earlier.

    .. code-block:: python

        >>> # Compute log-determinant of Aop
        >>> logdet(Aop, method='slq', gpu=True)
        141.52929878934194

    The above function call triggers the object `Aop` to automatically load the
    matrix data on the GPU.

    One could have used `A` instead of `Aop` in the above. However, an
    advantage of using `Aop` (instead of the matrix `A` directly) is that by
    calling the above `logdet` function (or another function) again on this
    matrix, the data of this matrix does not have to be re-allocated on the GPU
    device again. To highlight this point, call the above function again, but
    this time, set ``gram`` to `True` to compute something different.

    .. code-block:: python

        >>> # Compute log-determinant of Aop
        >>> logdet(Aop, method='slq', gpu=True, gram=True)
        141.52929878934194

    In the above example, no data is needed to be transferred from CPU host to
    GPU device again. However, if `A` was used instead of `Aop`, the data would
    have been transferred from CPU to GPU again for the second time. The `Aop`
    object holds the data on GPU for later use as long as this object does no
    go out of the scope of the python environment. Once the variable `Aop` goes
    out of scope, the matrix data on all the GPU devices will be cleaned
    automatically.

    **Affine matrix operator:**

    Use an object of :class:`imate.AffineMatrixFunction` to create the linear
    operator

    .. math::

        \\mathbf{A}(t) = \\mathbf{A} + t \\mathbf{I}.

    The object :math:`\\mathbf{A}(t)` can be passed to `logdet` function with
    multiple values for the parameter :math:`t` to compute their
    log-determinant all at once, as follows.

    .. code-block:: python
        :emphasize-lines: 8

        >>> # Import affine matrix function
        >>> from imate import toeplitz, logdet, AffineMatrixFunction

        >>> # Generate a sample matrix (a toeplitz matrix)
        >>> A = toeplitz(2, 1, size=100, gram=True)

        >>> # Create a matrix operator object from matrix A
        >>> Aop = AffineMatrixFunction(A)

        >>> # A list of parameters t to pass to Aop
        >>> t = [-1.0, 0.0, 1.0]

        >>> # Compute log-determinant of Aop for all parameters t
        >>> logdet(Aop, method='slq', parameters=t, min_num_samples=20,
        ...        max_num_samples=80, error_rtol=1e-3, confidence_level=0.95,
        ...        outlier_significance_level=0.001, plot=True, verbose=True)
        array([ 68.71411681, 135.88356906, 163.44156683])

    The output of the verbose argument is shown below. In the results section
    of the table below, each row `i` under the `inquiry` column corresponds to
    each element of the parameters ``t = [-1, 0, 1]`` that was specified by
    ``parameters`` argument.

    .. literalinclude:: ../_static/imate.logdet.slq-verbose-2.txt
        :language: python

    The output of the plot is shown below. Each colored curve corresponds to
    a parameter in ``t = [-1, 0, 1]``.

    .. image:: ../_static/images/plots/slq_convergence_2.png
        :align: center
        :width: 80%
        :class: custom-dark
    """

    # Checking input arguments
    error_atol, error_rtol = check_arguments(
            A, B, C, gram, p, assume_matrix, min_num_samples, max_num_samples,
            error_atol, error_rtol, confidence_level,
            outlier_significance_level, solver_tol, orthogonalize, num_threads,
            verbose, plot)

    # If the number of random vectors exceed the size of the vectors they
    # cannot be linearly independent and extra calculation with them will be
    # redundant.
    if A.shape[0] < max_num_samples:
        max_num_samples = A.shape[0]
    if A.shape[0] < min_num_samples:
        min_num_samples = A.shape[0]

    # Parallel processing
    if num_threads < 1:
        num_threads = multiprocessing.cpu_count()

    # Dispatch depending on 32-bit or 64-bit
    data_type_name = get_data_type_name(A)
    if data_type_name == b'float32':
        trace, error, num_outliers, samples, processed_samples_indices, \
                num_processed_samples, num_samples_used, converged, \
                tot_wall_time, alg_wall_time, cpu_proc_time = \
                _hutchinson_method_float(A, B, C, gram, p, assume_matrix,
                                         min_num_samples, max_num_samples,
                                         error_atol, error_rtol,
                                         confidence_level,
                                         outlier_significance_level,
                                         solver_tol, orthogonalize,
                                         num_threads)

    elif data_type_name == b'float64':
        trace, error, num_outliers, samples, processed_samples_indices, \
                num_processed_samples, num_samples_used, converged, \
                tot_wall_time, alg_wall_time, cpu_proc_time = \
                _hutchinson_method_double(A, B, C, gram, p, assume_matrix,
                                          min_num_samples, max_num_samples,
                                          error_atol, error_rtol,
                                          confidence_level,
                                          outlier_significance_level,
                                          solver_tol, orthogonalize,
                                          num_threads)
    else:
        raise TypeError('Data type should be either "float32" or "float64"')

    # Dictionary of output info
    info = {
        'matrix':
        {
            'data_type': data_type_name,
            'gram': gram,
            'exponent': p,
            'assume_matrix': assume_matrix,
            'size': A.shape[0],
            'sparse': isspmatrix(A),
            'nnz': get_nnz(A),
            'density': get_density(A),
            'num_inquiries': 1
        },
        'error':
        {
            'absolute_error': error,
            'relative_error': error / numpy.abs(trace),
            'error_atol': error_atol,
            'error_rtol': error_rtol,
            'confidence_level': confidence_level,
            'outlier_significance_level': outlier_significance_level
        },
        'convergence':
        {
            'converged': bool(converged),
            'min_num_samples': min_num_samples,
            'max_num_samples': max_num_samples,
            'num_samples_used': num_samples_used,
            'num_outliers': num_outliers,
            'samples': samples,
            'samples_mean': trace,
            'samples_processed_order': processed_samples_indices
        },
        'device':
        {
            'num_cpu_threads': num_threads,
            'num_gpu_devices': 0,
            'num_gpu_multiprocessors': 0,
            'num_gpu_threads_per_multiprocessor': 0
        },
        'time':
        {
            'tot_wall_time': tot_wall_time,
            'alg_wall_time': alg_wall_time,
            'cpu_proc_time': cpu_proc_time,
        },
        'solver':
        {
            'version': __version__,
            'orthogonalize': orthogonalize,
            'solver_tol': solver_tol,
            'method': 'hutchinson',
        }
    }

    # print summary
    if verbose:
        print_summary(info)

    # Plot results
    if plot:
        plot_convergence(info)

    if return_info:
        return trace, info
    else:
        return trace


# =======================
# hutchinson method float
# =======================

def _hutchinson_method_float(
        A,
        B,
        C,
        gram,
        p,
        assume_matrix,
        min_num_samples,
        max_num_samples,
        error_atol,
        error_rtol,
        confidence_level,
        outlier_significance_level,
        solver_tol,
        orthogonalize,
        num_threads):
    """
    This method processes single precision (32-bit) matrix ``A``.
    """

    vector_size = A.shape[0]

    # Allocate random array with Fortran ordering (first index is contiguous)
    # 2D array E should be treated as a matrix, random vectors are columns of E
    E = numpy.empty((vector_size, max_num_samples), dtype=numpy.float32,
                    order='F')

    # Get c pointer to E
    cdef float[::1, :] memoryview_E = E
    cdef float* cE = &memoryview_E[0, 0]

    init_tot_wall_time = time.perf_counter()
    init_cpu_proc_time = time.process_time()

    # Generate orthogonalized random vectors with unit norm
    generate_random_column_vectors[float](cE, vector_size, max_num_samples,
                                          int(orthogonalize), num_threads)

    samples = numpy.zeros((max_num_samples, ), dtype=numpy.float32)
    processed_samples_indices = numpy.zeros((max_num_samples, ), dtype=int)
    samples[:] = numpy.nan
    cdef int num_processed_samples = 0
    cdef int num_samples_used = 0
    cdef int converged = 0

    init_alg_wall_time = time.perf_counter()

    # Compute Gramian matrix if needed
    if gram:
        if (p == 1) and (B is None) and (C is None):
            # This special case does not needed the computation of Gramian.
            AtA = None
        else:
            AtA = A.T @ A
    else:
        AtA = None

    # Monte-Carlo sampling
    for i in range(max_num_samples):

        if converged == 0:

            # Stochastic estimator of trace using the i-th column of E
            samples[i] = _stochastic_trace_estimator_float(
                    A, AtA, B, C, E[:, i], gram, p, assume_matrix, solver_tol)

            # Store the index of processed samples
            processed_samples_indices[num_processed_samples] = i
            num_processed_samples += 1

            # Check whether convergence criterion has been met to stop.
            # This check can also be done after another parallel thread
            # set all_converged to "1", but we continue to update error.
            converged, num_samples_used = check_convergence(
                    samples, min_num_samples, processed_samples_indices,
                    num_processed_samples, confidence_level, error_atol,
                    error_rtol)

    alg_wall_time = time.perf_counter() - init_alg_wall_time

    trace, error, num_outliers = average_estimates(
            confidence_level, outlier_significance_level, max_num_samples,
            num_samples_used, processed_samples_indices, samples)

    tot_wall_time = time.perf_counter() - init_tot_wall_time
    cpu_proc_time = time.process_time() - init_cpu_proc_time

    return trace, error, num_outliers, samples, processed_samples_indices, \
        num_processed_samples, num_samples_used, converged, tot_wall_time, \
        alg_wall_time, cpu_proc_time


# ========================
# hutchinson method double
# ========================

def _hutchinson_method_double(
        A,
        B,
        C,
        gram,
        p,
        assume_matrix,
        min_num_samples,
        max_num_samples,
        error_atol,
        error_rtol,
        confidence_level,
        outlier_significance_level,
        solver_tol,
        orthogonalize,
        num_threads):
    """
    This method processes double precision (64-bit) matrix ``A``.
    """

    vector_size = A.shape[0]

    # Allocate random array with Fortran ordering (first index is contiguous)
    # 2D array E should be treated as a matrix, random vectors are columns of E
    E = numpy.empty((vector_size, max_num_samples), dtype=numpy.float64,
                    order='F')

    # Get c pointer to E
    cdef double[::1, :] memoryview_E = E
    cdef double* cE = &memoryview_E[0, 0]

    init_tot_wall_time = time.perf_counter()
    init_cpu_proc_time = time.process_time()

    # Generate orthogonalized random vectors with unit norm
    generate_random_column_vectors[double](cE, vector_size, max_num_samples,
                                           int(orthogonalize), num_threads)

    samples = numpy.zeros((max_num_samples, ), dtype=numpy.float64)
    processed_samples_indices = numpy.zeros((max_num_samples, ), dtype=int)
    samples[:] = numpy.nan
    cdef int num_processed_samples = 0
    cdef int num_samples_used = 0
    cdef int converged = 0

    init_alg_wall_time = time.perf_counter()

    # Compute Gramian matrix if needed
    if gram:
        if (p == 1) and (B is None) and (C is None):
            # This special case does not needed the computation of Gramian.
            AtA = None
        else:
            AtA = A.T @ A
    else:
        AtA = None

    # Monte-Carlo sampling
    for i in range(max_num_samples):

        if converged == 0:

            # Stochastic estimator of trace using the i-th column of E
            samples[i] = _stochastic_trace_estimator_double(
                    A, AtA, B, C, E[:, i], gram, p, assume_matrix, solver_tol)

            # Store the index of processed samples
            processed_samples_indices[num_processed_samples] = i
            num_processed_samples += 1

            # Check whether convergence criterion has been met to stop.
            # This check can also be done after another parallel thread
            # set all_converged to "1", but we continue to update error.
            converged, num_samples_used = check_convergence(
                    samples, min_num_samples, processed_samples_indices,
                    num_processed_samples, confidence_level, error_atol,
                    error_rtol)

    alg_wall_time = time.perf_counter() - init_alg_wall_time

    trace, error, num_outliers = average_estimates(
            confidence_level, outlier_significance_level, max_num_samples,
            num_samples_used, processed_samples_indices, samples)

    tot_wall_time = time.perf_counter() - init_tot_wall_time
    cpu_proc_time = time.process_time() - init_cpu_proc_time

    return trace, error, num_outliers, samples, processed_samples_indices, \
        num_processed_samples, num_samples_used, converged, tot_wall_time, \
        alg_wall_time, cpu_proc_time


# ================================
# stochastic trace estimator float
# ================================

cdef float _stochastic_trace_estimator_float(
        A,
        AtA,
        B,
        C,
        E,
        gram,
        p,
        assume_matrix,
        solver_tol) except *:
    """
    Stochastic trace estimator based on set of vectors E and AinvpE.

    :param E: Set of random vectors of shape ``(vector_size, num_vectors)``.
        Note this is Fortran ordering, meaning that the first index is
        contiguous. Hence, to call the i-th vector, use ``&E[0][i]``.
        Here, iteration over the first index is continuous.
    :type E: cython memoryview (float)

    :param AinvpE: Set of random vectors of the same shape as ``E``.
        Note this is Fortran ordering, meaning that the first index is
        contiguous. Hence, to call the i-th vector, use ``&AinvpE[0][i]``.
        Here, iteration over the first index is continuous.
    :type AinvpE: cython memoryview (float)

    :param num_vectors: Number of columns of vectors array.
    :type num_vectors: int

    :param vector_size: Number of rows of vectors array.
    :type vector_size: int

    :param num_parallel_threads: Number of OpenMP parallel threads
    :type num_parallel_threads: int

    :return: Trace estimation.
    :rtype: float
    """

    # Check AtA is not None when AtA is needed
    if gram and (AtA is None):
        if not ((p == 1) and (B is None) and (C is None)):
            raise RuntimeError('"AtA" cannot be None.')

    # Multiply operator * B * E
    OpE = _operator_dot(A, AtA, p, gram, assume_matrix, solver_tol, B, E)

    # Multiply operator * C * OpE
    if C is not None:
        OpE = _operator_dot(A, AtA, p, gram, assume_matrix, solver_tol, C, OpE)

    # Get c pointer to E
    cdef float[:] memoryview_E = E
    cdef float* cE = &memoryview_E[0]

    # Get c pointer to OpE.
    cdef float[:] memoryview_OpE = OpE
    cdef float* cOpE = &memoryview_OpE[0]

    # Inner product of E and OpE
    cdef int vector_size = A.shape[0]
    cdef float inner_prod

    if gram and (numpy.abs(p) == 1) and (B is None) and (C is None):
        inner_prod = cVectorOperations[float].inner_product(cOpE, cOpE,
                                                            vector_size)
    else:
        inner_prod = cVectorOperations[float].inner_product(cE, cOpE,
                                                            vector_size)

    # Hutcinson trace estimate
    cdef float trace_estimate = vector_size * inner_prod

    return trace_estimate


# =================================
# stochastic trace estimator double
# =================================

cdef double _stochastic_trace_estimator_double(
        A,
        AtA,
        B,
        C,
        E,
        gram,
        p,
        assume_matrix,
        solver_tol) except *:
    """
    Stochastic trace estimator based on set of vectors E and AinvpE.

    :param E: Set of random vectors of shape ``(vector_size, num_vectors)``.
        Note this is Fortran ordering, meaning that the first index is
        contiguous. Hence, to call the i-th vector, use ``&E[0][i]``.
        Here, iteration over the first index is continuous.
    :type E: cython memoryview (double)

    :param AinvpE: Set of random vectors of the same shape as ``E``.
        Note this is Fortran ordering, meaning that the first index is
        contiguous. Hence, to call the i-th vector, use ``&AinvpE[0][i]``.
        Here, iteration over the first index is continuous.
    :type AinvpE: cython memoryview (double)

    :param num_vectors: Number of columns of vectors array.
    :type num_vectors: int

    :param vector_size: Number of rows of vectors array.
    :type vector_size: int

    :param num_parallel_threads: Number of OpenMP parallel threads
    :type num_parallel_threads: int

    :return: Trace estimation.
    :rtype: double
    """

    # Check AtA is not None when AtA is needed
    if gram and (AtA is None):
        if not ((p == 1) and (B is None) and (C is None)):
            raise RuntimeError('"AtA" cannot be None.')

    # Multiply operator * B * E
    OpE = _operator_dot(A, AtA, p, gram, assume_matrix, solver_tol, B, E)

    # Multiply operator * C * OpE
    if C is not None:
        OpE = _operator_dot(A, AtA, p, gram, assume_matrix, solver_tol, C, OpE)

    # Get c pointer to E
    cdef double[:] memoryview_E = E
    cdef double* cE = &memoryview_E[0]

    # Get c pointer to OpE.
    cdef double[:] memoryview_OpE = OpE
    cdef double* cOpE = &memoryview_OpE[0]

    # Inner product of E and OpE
    cdef int vector_size = A.shape[0]
    cdef double inner_prod

    if gram and (numpy.abs(p) == 1) and (B is None) and (C is None):
        inner_prod = cVectorOperations[double].inner_product(cOpE, cOpE,
                                                             vector_size)
    else:
        inner_prod = cVectorOperations[double].inner_product(cE, cOpE,
                                                             vector_size)

    # Hutcinson trace estimate
    cdef double trace_estimate = vector_size * inner_prod

    return trace_estimate


# ============
# operator dot
# ============

def _operator_dot(A, AtA, p, gram, assume_matrix, solver_tol, B, E):
    """
    Computes either of the followings:

    * Ainv * B * E
    * (Ainv ** p) * B * E
    * AtA * B * E
    * (AtA ** p) * B * E
    """

    # Multiply B by E
    if B is not None:
        BE = B @ E
    else:
        # Assume B is identity matrix
        BE = E

    # In the following, OpE is the action of the operator A**(-p) to the
    # vector BE. The exponent "p" is the "p" argument which is default
    # to one. Ainv means the inverse of A.
    if p == 0:
        # Ainvp is the identity matrix
        OpE = BE

    elif p == 1:
        # Perform inv(A) * BE. This requires GIL
        if gram:
            if B is None:
                OpE = linear_solver(A.T, BE, assume_matrix, solver_tol)
            else:
                OpE = linear_solver(AtA, BE, assume_matrix, solver_tol)
        else:
            OpE = linear_solver(A, BE, assume_matrix, solver_tol)

    elif p > 1:
        # Perform Ainv * Ainv * ... Ainv * BE where Ainv is repeated p times
        # where p is the exponent.
        OpE = BE

        if gram:
            for i in range(p):
                OpE = linear_solver(AtA, OpE, assume_matrix, solver_tol)
        else:
            for i in range(p):
                OpE = linear_solver(A, OpE, assume_matrix, solver_tol)

    elif p == -1:
        # Performing Ainv**(-1) BE, where Ainv**(-1) it A itself.
        OpE = A @ BE

    elif p < -1:
        # Performing Ainv**(-p) * BE where Ainv**(-p) = A**p.
        AinvpE = BE
        if gram:
            for i in range(numpy.abs(p)):
                OpE = AtA @ OpE
        else:
            for i in range(numpy.abs(p)):
                OpE = A @ OpE

    return OpE
