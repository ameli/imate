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

from ._eigenvalue_method import eigenvalue_method
from ._slq_method import slq_method


# =========
# trlinfrac
# =========

def trlinfrac(
        A,
        gram=False,
        p=1.0,
        coeff=[1.0, 0.0, 1.0, 1.0],
        return_info=False,
        method='eigenvalue',
        **options):
    """
    Trace of the exponential of matrix or linear operator.

    Given the matrix or the linear operator :math:`\\mathbf{A}` and the real
    exponent :math:`p`, the following is computed:

    .. math::

        \\mathrm{trace} f \\left(\\mathbf{A}^p \\right),

    where the function :math:`f` is defined by

    .. math::

        f(\\lambda) = \\frac{a \\lambda + b}{c \\lambda +d},

    provided that :math:`ad - bc \\neq 0`.

    If ``gram`` is `True`, then :math:`\\mathbf{A}` in the above is replaced by
    the Gramian matrix :math:`\\mathbf{A}^{\\intercal} \\mathbf{A}`, and the
    following is instead computed:

    .. math::

        \\mathrm{trace} f
        \\left((\\mathbf{A}^{\\intercal}\\mathbf{A})^p \\right).

    If :math:`\\mathbf{A} = \\mathbf{A}(t)` is a linear operator of the class
    :class:`imate.AffineMatrixFunction` with the parameter :math:`t`, then for
    an input  tuple :math:`t = (t_1, \\dots, t_q)`, an array output of the size
    :math:`q` is returned, namely:

    .. math::

        \\mathrm{trace} f \\left((\\mathbf{A}(t_i))^p \\right),
        \\quad i=1, \\dots, q.

    Parameters
    ----------

    A : numpy.ndarray, scipy.sparse, :class:`imate.Matrix`, or \
            :class:`imate.AffineMatrixFunction`
        A sparse or dense matrix or linear operator. The linear
        operators :class:`imate.Matrix` and :class:`imate.AffineMatrixFunction`
        can be used only if ``method=slq``. See details in
        :ref:`slq method <imate.trlinfrac.slq>`. If ``method=cholesky``, the
        matrix `A` should be positive-definite. If ``method=slq`` and
        ``gram=False``, the input matrix `A` should be symmetric. If
        ``gram=True``, the input matrix can be non-square.

    gram : bool, default=False
        If `True`, the trace-exponential of the Gramian matrix,
        :math:`(\\mathbf{A}^{\\intercal}\\mathbf{A})^p`, is computed. The
        Gramian matrix itself is not directly computed. If `False`, the
        trace-exponential of :math:`\\mathbf{A}^p` is computed.

    p : float, default=1.0
        The real exponent :math:`p` in :math:`\\mathbf{A}^p`.

    coeff : (tuple, list, numpy.array), default=[1,0, 0.0, 1.0, 1.0]
        A list, tuple, or array of four float numbers representing the
        coefficients :math:`a`, :math:`b`, :math:`c`, and :math:`d`,
        respectively.

    return_info : bool, default=False
        If `True`, this function also returns a dictionary containing
        information about the inner computation, such as process time,
        algorithm settings, etc. See the documentation for each `method` for
        details.

    method : {'eigenvalue', 'slq'}, default='eigenvalue'
        The method of computing trace-exponential. See documentation for each
        method:

        * :ref:`eigenvalue <imate.trlinfrac.eigenvalue>`
        * :ref:`slq <imate.trlinfrac.slq>`

    options : `**kwargs`
        Extra arguments that are specific to each method. See the documentation
        for each `method` for details.

    Returns
    -------

    trlinfrac : float or numpy.array
        Trace-exponential of matrix. If ``method=slq`` and if `A` is of type
        :class:`imate.AffineMatrixFunction` with an array of ``parameters``,
        then the output is an array.

    info : dict
        (Only if ``return_info`` is `True`) A dictionary of information with at
        least the following keys. Further keys specific to each method can be
        found in the documentation of each method.

        * ``matrix``:
            * ``data_type``: `str`, {`float32`, `float64`, `float128`}, type of
              the matrix data.
            * ``gram``: `bool`, whether the matrix `A` or its Gramian is
              considered.
            * ``exponent``: `float`, the exponent `p` in :math:`\\mathbf{A}^p`.
            * ``size``: `int`, The size of matrix `A`.
            * ``sparse``: `bool`, whether the matrix `A` is sparse or dense.
            * ``nnz``: `int`, if `A` is sparse, the number of non-zero elements
              of `A`.
            * ``density``: `float`, if `A` is sparse, the density of `A`, which
              is the `nnz` divided by size squared.
            * ``num_inquiries``: `int`, The size of inquiries of each parameter
              of the linear operator `A`. If `A` is a matrix, this is always
              `1`. For more details see :ref:`slq method
              <imate.trlinfrac.slq>`.

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
            * ``cpu_proc_time``: `float`, CPU processing time of computation.

        * ``solver``:
            * ``version``: `str`, version of imate.
            * ``method``: `str`, method of computation.

    Raises
    ------

    ImportError
        If the package has not been compiled with GPU support, but ``gpu`` is
        `True`. Either set ``gpu`` to `False` to use the existing installed
        package. Alternatively, export the environment variable ``USE_CUDA=1``
        and recompile the source code of the package.

    See Also
    --------

    imate.logdet
    imate.trace
    imate.trexp
    imate.traceinv
    imate.schatten

    Notes
    -----

    **Method of Computation:**

    See documentation for each method below.

    * :ref:`eigenvalue <imate.trlinfrac.eigenvalue>`: uses spectral
      decomposition.
      Suitable for small matrices (:math:`n < 2^{12}`). The solution is exact.
    * :ref:`slq <imate.trlinfrac.slq>`: uses stochastic Lanczos quadrature
      (SLQ), which is a randomized algorithm. Can be used on very large
      matrices (:math:`n > 2^{12}`). The solution is an approximation.

    **Input Matrix:**

    The input `A` can be either of:

    * A matrix, such as `numpy.ndarray`, or `scipy.sparse`.
    * A linear operator representing a matrix using :class:`imate.Matrix` (
      only if ``method=slq``).
    * A linear operator representing a one-parameter family of an affine matrix
      function :math:`t \\mapsto \\mathbf{A} + t\\mathbf{B}`, using
      :class:`imate.AffineMatrixFunction` (only if ``method=slq``).

    **Output:**

    The output is a scalar. However, if `A` is the linear operator of the type
    :class:`imate.AffineMatrixFunction` representing the matrix function
    :math:`\\mathbf{A}(t) = \\mathbf{A} + t \\mathbf{B}`, then if the parameter
    :math:`t` is given as the tuple :math:`t = (t_1, \\dots, t_q)`, then the
    output of this function is an array of size :math:`q` corresponding to the
    trace-exponential of each :math:`\\mathbf{A}(t_i)`.

    .. note::

        When `A` represents
        :math:`\\mathbf{A}(t) = \\mathbf{A} + t \\mathbf{I}`, where
        :math:`\\mathbf{I}` is the identity matrix, and :math:`t` is given by
        a tuple :math:`t = (t_1, \\dots, t_q)`, by setting ``method=slq``, the
        computational cost of an array output of size `q` is the same as
        computing for a single :math:`t_i`. Namely, the trace-exponential of
        only :math:`\\mathbf{A}(t_1)` is computed, and the trace-exponential of
        the rest of :math:`q=2, \\dots, q` are obtained from the result of
        :math:`t_1` immediately.

    Examples
    --------

    **Sparse matrix:**

    Compute the trace-exponential of a sample sparse Toeplitz matrix created by
    :func:`imate.toeplitz` function.

    .. code-block:: python

        >>> # Import packages
        >>> from imate import toeplitz, trlinfrac

        >>> # Generate a sample matrix (a toeplitz matrix)
        >>> A = toeplitz(2, 1, size=100)

        >>> # Compute trace-exponential with eigenvalue method (default method)
        >>> trlinfrac(A)
        138.6294361119891

    Alternatively, compute the trace-exponential of
    :math:`\\mathbf{A}^{\\intercal} \\mathbf{A}`:

    .. code-block:: python

        >>> # Compute trace-exponential of the Gramian to the power of 3:
        >>> trlinfrac(A, p=3, gram=True)
        831.7766166719346

    **Output information:**

    Print information about the inner computation:

    .. code-block:: python

        >>> ld, info = trlinfrac(A, return_info=True)
        >>> print(ld)
        138.6294361119891

        >>> # Print dictionary neatly using pprint
        >>> from pprint import pprint
        >>> pprint(info)
        {
            'matrix': {
                'data_type': b'float64',
                'density': 0.0298,
                'exponent': 1.0,
                'gram': False,
                'nnz': 298,
                'num_inquiries': 1,
                'size': 100,
                'sparse': True
            },
            'solver': {
                'cholmod_used': True,
                'method': 'eigenvalue',
                'version': '0.13.0'
            },
            'device': {
                'num_cpu_threads': 8,
                'num_gpu_devices': 0,
                'num_gpu_multiprocessors': 0,
                'num_gpu_threads_per_multiprocessor': 0
            },
            'time': {
                'alg_wall_time': 0.000903537031263113,
                'cpu_proc_time': 0.0010093420000032438,
                'tot_wall_time': 0.000903537031263113
            }
        }

    **Large matrix:**

    Compute trace-exponential of a very large sparse matrix using `SLQ` method.
    This method does not compute race-exponential exactly, rather, the result
    is an approximation using Monte-Carlo sampling. The following example uses
    at least `100` samples.

    .. code-block:: python

        >>> # Generate a matrix of size one million
        >>> A = toeplitz(2, 1, size=1000000, gram=True)

        >>> # Approximate trace-exponential using stochastic Lanczos quadrature
        >>> # with at least 100 Monte-Carlo sampling
        >>> ld, info = trlinfrac(A, method='slq', min_num_samples=100,
        ...                      max_num_samples=200, return_info=True)
        >>> print(ld)
        1386187.5751816272

        >>> # Find the time it took to compute the above
        >>> print(info['time'])
        {
            'tot_wall_time': 15.908390650060028,
            'alg_wall_time': 15.890228271484375,
            'cpu_proc_time': 116.93080989600001
        }

    **Matrix operator:**

    The following example uses an object of :class:`imate.Matrix`. Note that
    this can be only applied to ``method=slq``. See further details in
    :ref:`slq method <imate.trlinfrac.slq>`.

    .. code-block:: python
        :emphasize-lines: 8

        >>> # Import matrix operator
        >>> from imate import toeplitz, trlinfrac, Matrix

        >>> # Generate a sample matrix (a toeplitz matrix)
        >>> A = toeplitz(2, 1, size=100, gram=True)

        >>> # Create a matrix operator object from matrix A
        >>> Aop = Matrix(A)

        >>> # Compute trace-exponential of Aop
        >>> trlinfrac(Aop, method='slq')
        141.52929878934194

    **Affine matrix operator:**

    The following example uses an object of
    :class:`imate.AffineMatrixFunction` to create the linear operator:

    .. math::

        t \\mapsto \\mathbf{A} + t \\mathbf{I}

    Note that this can be only applied to ``method=slq``. See further details
    in :ref:`slq method <imate.trlinfrac.slq>`.

    .. code-block:: python
        :emphasize-lines: 8

        >>> # Import affine matrix function
        >>> from imate import toeplitz, trlinfrac, AffineMatrixFunction

        >>> # Generate a sample matrix (a toeplitz matrix)
        >>> A = toeplitz(2, 1, size=100, gram=True)

        >>> # Create a matrix operator object from matrix A
        >>> Aop = AffineMatrixFunction(A)

        >>> # A list of parameters t to pass to Aop
        >>> t = [-1.0, 0.0, 1.0]

        >>> # Compute trace-exp of Aop for all parameters t
        >>> trlinfrac(Aop, method='slq', parameters=t)
        array([ 68.71411681, 135.88356906, 163.44156683])
    """

    if method == 'eigenvalue':
        return eigenvalue_method(A, gram=gram, p=p, coeff=coeff,
                                 return_info=return_info, **options)

    elif method == 'slq':
        return slq_method(A, gram=gram, p=p, coeff=coeff,
                          return_info=return_info, **options)

    else:
        raise ValueError('Method "%s" is invalid.' % (method))
