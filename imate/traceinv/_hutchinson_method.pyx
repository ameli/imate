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
    Trace of matrix or linear operator using stochastic Lanczos quadrature
    method.

    If `C` is `None`, given the matrices :math:`\\mathbf{A}` and
    :math:`\\mathbf{B}` and the integer exponent :math:`p`, the following is
    computed:

    .. math::

        \\mathrm{trace} \\left(\\mathbf{B} \\mathbf{A}^{-p} \\right).

    If `B` is `None`, it is assumed that :math:`\\mathbf{B}` is the identity
    matrix.

    If `C` is not `None`, given the matrix :math:`\\mathbf{C}`, the following
    is instead computed:

    .. math::

        \\mathrm{trace} \\left(\\mathbf{B} \\mathbf{A}^{-p} \\mathbf{C}
        \\mathbf{A}^{-p} \\right).

    If ``gram`` is `True`, then :math:`\\mathbf{A}` in the above is replaced by
    the Gramian matrix :math:`\\mathbf{A}^{\\intercal} \\mathbf{A}`. Namely, if
    `C` is `None`:

    .. math::

        \\mathrm{trace} \\left(\\mathbf{B}
        (\\mathbf{A}^{\\intercal}\\mathbf{A})^{-p} \\right),

    and if `C` is not `None`,

    .. math::

        \\mathrm{trace} \\left(\\mathbf{B}
        (\\mathbf{A}^{\\intercal}\\mathbf{A})^{-p} \\mathbf{C}
        (\\mathbf{A}^{\\intercal}\\mathbf{A})^{-p} \\right).

    Parameters
    ----------

    A : numpy.ndarray, scipy.sparse
        A sparse or dense matrix. If ``gram`` is `True`, the matrix can be
        non-square.

        .. note::

            In the Hutchinson method, the matrix cannot be a type of
            :class:`Matrix` or :class:`imate.AffineMatrixFunction` classes.

    gram : bool, default=False
        If `True`, the trace of the Gramian matrix,
        :math:`(\\mathbf{A}^{\\intercal}\\mathbf{A})^p`, is computed. The
        Gramian matrix itself is not directly computed. If `False`, the
        trace of :math:`\\mathbf{A}^p` is computed.

    p : float, default=1.0
        The integer exponent :math:`p` in :math:`\\mathbf{A}^{-p}`.

    return_info : bool, default=False
        If `True`, this function also returns a dictionary containing
        information about the inner computation, such as process time,
        algorithm settings, etc.

    B : numpy.ndarray, scipy.sparse
        A sparse or dense matrix. `B` should be the same size and type of `A`.
        if `B` is `None`, it is assumed that `B` is the identity matrix.

    C : numpy.ndarray, scipy.sparse
        A sparse or dense matrix. `C` should be the same size and type of `A`.

    assume_matrix : str {'gen', 'sym', 'pos', 'sym_pos'}, default: 'gen'
        Type of matrix `A`:

        * ``gen`` assumes `A` is a generic matrix.
        * ``sym`` assumes `A` is symmetric.
        * ``pos`` assumes `A` is positive-definite.
        * ``sym_pos`` assumes `A` is symmetric and positive-definite.

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

    solver_tol : float, default=1e-6
        Tolerance of solving linear system.

    orthogonalize : int, default=0
        If `True`, it orthogonalizes the set of random vectors used for
        Monte-Carlo sampling. This might lead to a better estimation of the
        output.

    num_threads : int, default=0
        Number of processor threads to employ for parallel computation on CPU.
        If set to `0` or a number larger than the available number of threads,
        all threads of the processor are used. The parallelization is performed
        over the Monte-Carlo iterations.

    verbose : bool, default=False
        Prints extra information about the computations.

    plot : bool, default=False
        Plots convergence of samples. For this, the packages `matplotlib` and
        `seaborn` should be installed. If no display is available (such as
        running this code on remote machines), the plots are saved as an `SVG`
        file in the current directory.

    Returns
    -------

    traceinv : float or numpy.array
        Trace of inverse of matrix.

    info : dict
        (Only if ``return_info`` is `True`) A dictionary of information with
        the following.

        * ``matrix``:
            * ``data_type``: `str`, {`float32`, `float64`, `float128`}. Type of
              the matrix data.
            * ``gram``: `bool`, whether the matrix `A` or its Gramian is
              considered.
            * ``exponent``: `float`, the exponent `p` in :math:`\\mathbf{A}^p`.
            * ``assume_matrix``: `str`, {`gen`, `sym`, `pos`, `sym_pos`},
              determines the type of matrix `A`.
            * ``size``: (int, int) The size of matrix `A`.
            * ``sparse``: `bool`, whether the matrix `A` is sparse or dense.
            * ``nnz``: `int`, if `A` is sparse, the number of non-zero elements
              of `A`.
            * ``density``: `float`, if `A` is sparse, the density of `A`, which
              is the `nnz` divided by size squared.
            * ``num_inquiries``: `int`, the size of inquiries of each parameter
              of the linear operator `A`. If `A` is a matrix, this is always
              `1`. If `A` is a type of :class:`AffineMatrixFunction`, this
              value is the number of :math:`t_i` parameters.

        * ``convergence``:
            * ``converged``: `bool`, whether the Monte-Carlo sampling
              converged.
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
            * ``method``: 'hutchinson'.
            * ``solver_tol``: `float`, tolerance of solving linear system.
            * ``orthogonalize``: `bool`, orthogonalization flag.

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

    imate.logdet
    imate.trace
    imate.schatten

    Notes
    -----

    **Computational Complexity:**

    This method uses the Hutchinson, which is a randomized algorithm. The
    computational complexity of this method is

    .. math::

        \\mathcal{O}((\\rho n^2s),

    where :math:`n` is the matrix size, :math:`\\rho` is the density of
    sparse matrix (for dense matrix, :math:`\\rho=1`), and :math:`s` is the
    number of samples (set with ``min_num_samples`` and ``max_num_samples``).

    This method can be used on very large matrices (:math:`n > 2^{12}`). The
    solution is an approximation.

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

    **Basic Usage:**

    Compute the trace of :math:`\\mathbf{A}^{-2}`:

    .. code-block:: python

        >>> # Import packages
        >>> from imate import toeplitz, traceinv

        >>> # Generate a sample matrix
        >>> A = toeplitz(2, 1, size=100)

        >>> # Compute trace of inverse
        >>> traceinv(A, p=2, method='hutchinson')
        24.73726368966402

    Compute the trace of :math:`(\\mathbf{A}^{\\intercal} \\mathbf{A})^{-2}`:

    .. code-block:: python

        >>> # Using Gramian matrix of A
        >>> traceinv(A, gram=True, p=2, method='hutchinson')
        17.751659383784748

    Compute the trace of :math:`\\mathbf{B} \\mathbf{A}^{-2}`:

    .. code-block:: python

        >>> # Generate another sample matrix
        >>> B = toeplitz(4, 3, size=100)

        >>> # Using Gramian matrix of A
        >>> traceinv(A, p=2, method='hutchinson', B=B)
        99.8817360381704

    Compute the trace of :math:`\\mathbf{B} \\mathbf{A}^{-2} \\mathbf{C}
    \\mathbf{A}^{-2}`:

    .. code-block:: python

        >>> # Generate another sample matrix
        >>> C = toeplitz(5, 4, size=100)

        >>> # Using Gramian matrix of A
        >>> traceinv(A, p=2, method='hutchinson', B=B, C=C)
        124.45436379980006

    Compute the trace of :math:`\\mathbf{B} (\\mathbf{A}^{\\intercal}
    \\mathbf{A})^{-2} \\mathbf{C} (\\mathbf{A}^{\\intercal} \\mathbf{A})^{-2}`:

    .. code-block:: python

        >>> # Using Gramian matrix of A
        >>> traceinv(A, gram=True, p=2, method='hutchinson', B=B, C=C)
        5.517453125230929

    **Verbose output:**

    By setting ``verbose`` to `True`, useful info about the process is
    printed.

    .. literalinclude:: ../_static/data/imate.traceinv.hutchinson-verbose.txt
        :language: python

    **Output information:**

    Print information about the inner computation:

    .. code-block:: python

        >>> ti, info = traceinv(A, method='hutchinson', return_info=True)
        >>> print(ti)
        50.059307947603585

        >>> # Print dictionary neatly using pprint
        >>> from pprint import pprint
        >>> pprint(info)
        {
            'matrix': {
                'assume_matrix': 'gen',
                'data_type': b'float64',
                'density': 0.0199,
                'exponent': 1,
                'gram': False,
                'nnz': 199,
                'num_inquiries': 1,
                'size': (100, 100),
                'sparse': True
            },
            'convergence': {
                'converged': False,
                 'max_num_samples': 50,
                 'min_num_samples': 10,
                 'num_outliers': 0,
                 'num_samples_used': 50,
                 'samples': array([52.237154, ..., 51.37932704]),
                 'samples_mean': 50.059307947603585,
                 'samples_processed_order': array([ 0, ..., 49])
            },
            'error': {
                'absolute_error': 0.8111131801161796,
               'confidence_level': 0.95,
               'error_atol': 0.0,
               'error_rtol': 0.01,
               'outlier_significance_level': 0.001,
               'relative_error': 0.016203044216375525
            },
            'solver': {
                'method': 'hutchinson',
                'orthogonalize': True,
                'solver_tol': 1e-06,
                'version': '0.16.0'
            },
            'device': {
                'num_cpu_threads': 8,
                'num_gpu_devices': 0,
                'num_gpu_multiprocessors': 0,
                'num_gpu_threads_per_multiprocessor': 0
            },
            'time': {
                'alg_wall_time': 0.03236744087189436,
              'cpu_proc_time': 0.047695197999999994,
              'tot_wall_time': 0.033352302853018045
            }
        }

    **Large matrix:**

    Compute the trace of :math:`\\mathbf{A}^{-1}` for a very large sparse
    matrix using at least `100` samples.

    .. code-block:: python
        :emphasize-lines: 5, 6, 7

        >>> # Create a symmetric positive-definite matrix of size one million.
        >>> A = toeplitz(2, 1, size=1000000, gram=True)

        >>> # Approximate trace using hutchinson method
        >>> ti, info = traceinv(A, method='hutchinson', solver_tol=1e-4,
        ...                     assume_matrix='sym_pos', min_num_samples=100,
        ...                     max_num_samples=200, return_info=True)
        >>> print(ti)
        333292.3226031165

        >>> # Find the time it took to compute the above
        >>> print(info['time'])
        {
            'tot_wall_time': 175.93423152901232,
            'alg_wall_time': 119.86316476506181,
            'cpu_proc_time': 572.180877451
        }

    Compare the result of the above approximation with the exact solution of
    the trace using the analytic relation for Toeplitz matrix. See
    :func:`imate.sample_matrices.toeplitz_traceinv` for details.

    .. code-block:: python

        >>> from imate.sample_matrices import toeplitz_traceinv
        >>> toeplitz_traceinv(2, 1, size=1000000, gram=True)
        333333.2222222222

    It can be seen that the error of approximation is :math:`0.012 \\%`. This
    accuracy is remarkable considering that the computation on such a large
    matrix took on 119 seconds. Computing the trace of such a large matrix
    using any of the exact methods (such as ``exact`` or ``eigenvalue``) is
    infeasible.

    **Plotting:**

    By setting ``plot`` to `True`, plots of samples during Monte-Carlo
    iterations and the convergence of their mean are generated.

    .. code-block:: python

        >>> A = toeplitz(2, 1, size=1000000, gram=True)
        >>> traceinv(A, method='hutchinson', assume_matrix='sym_pos',
        ...          solver_tol=1e-4, min_num_samples=50, max_num_samples=150,
        ...          error_rtol=2e-4, confidence_level=0.95,
        ...          outlier_significance_level=0.001, plot=True)

    .. image:: ../_static/images/plots/traceinv_hutchinson_convergence.png
        :align: center
        :class: custom-dark

    In the left plot, the samples are shown in circles and the cumulative mean
    of the samples is shown by a solid black curve. The shaded area corresponds
    to the 95% confidence interval :math:`\\pm 1.96 \\sigma`, which is set by
    ``confidence_level=0.95``. The samples outside the interval of 99.9% are
    considered outliers, which is set by the significance level
    ``outlier_significance_level=0.001``.

    In the right plot, the darker shaded area in the interval :math:`[0, 50]`
    shows the minimum number of samples and is set by ``min_num_samples=50``.
    The iterations do not stop till the minimum number of iterations is passed.
    We can observe that sampling is terminated after 140 iterations where the
    relative error of samples reaches 0.02% since we set ``error_rtol=2e-4``.
    The lighter shaded area in the interval :math:`[140, 150]` corresponds to
    the iterations that were not performed to reach the specified maximum
    iterations by ``max_num_samples=150``.
    """

    # Checking input arguments
    error_atol, error_rtol, square = check_arguments(
            A, B, C, gram, p, return_info, assume_matrix, min_num_samples,
            max_num_samples, error_atol, error_rtol, confidence_level,
            outlier_significance_level, solver_tol, orthogonalize, num_threads,
            verbose, plot)

    # If the number of random vectors exceed the size of the vectors they
    # cannot be linearly independent and extra calculation with them will be
    # redundant.
    if A.shape[1] < max_num_samples:
        max_num_samples = A.shape[1]
    if A.shape[1] < min_num_samples:
        min_num_samples = A.shape[1]

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
            'size': A.shape,
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
