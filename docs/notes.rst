*****
Notes
*****

Some notes to myself when completing the documentation later.

* It seems there is a difference of results when gpu is enabled or disabled.
  When the data is double precision, this difference is very small, almost
  unimportant.But for single precision data, the difference of cublas and cpu
  based computation for float precision is more pronounced. In particular, the
  Lanczos iterations quickly lose their orthogonality. This issue is in
  particular more observed when single precision data ``float32``` is used. To
  avoid this, either use double precision, or enable *reorthogonalization* in
  Lancozs process. 

* The ``cusparse`` documentation here:
  https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spmv
  says that:

      "The routine requires extra storage for CSR format (all algorithms) and
      for COO format with CUSPARSE_SPMM_COO_ALG2 algorithm."

  However, it seems the ``this->buffer_size`` (see ``cu_csr_matrix.cu``) is
  always zero even for various algorithms (currently I use
  ``CUSPARSE_MV_ALG_DEFAULT``, but for any other algorithm, the buffer size
  is still zero).

  If I can be certain that the buffer size is "always" zero for the algorithm
  that I use, then I do not need to call ``this->allocate_buffer`` from the
  ``this->dot()``function. Previously, the ``dot()`` function had ``const``
  at the end of its signature. However, because we call ``allocate_buffer``,
  and it changes a member data, the ``dot()`` function for this class and
  ALL of its base classes, and even the other non-cuda classes that are
  derived from the ``cLinearOperator`` class has to be non ``const`` member
  functions.

  If I can be certain that the buffer size is always zero, I can return back
  the constness to these functions and do not call the allocate buffer.
  Or, I can call allocate buffer to compute the buffer size, but do not
  allocate it. In this case, I should do ``assert(buffer_size==0)`` just in
  case if in some applications it has to be non-zero buffer.

  The ``const``s that I removed:

  1. In ``cuCSRAffineMatrixFunction`` and ``cuCSCAffineMatrixFunction``, the
     member data ``cuCSRMatrix<DataType> A, B``, and
     ``cuCSCMatrix<DataType>A, B`` were ``const`` member data before. Now, they
     are not const.
  2. In ``cu_lanczos_tridiagonalization`` and
     ``cu_golub_kahn_bidiagonalization``, all ``cuLinearOperator`` arguments
     were ``const``. Now they are not const.
  3. All functions of ``dot``, ``dot_plus``, ``transpose_dot``, and
     ``transpose_dot_plus`` were const member functions for the entire
     classes of ``cLinearOperator`` and ``cuLinearOperator`` were const
     functions. Now they are not const.


====
TODO
====

* other functions (besides traceinv and logdet)
* interpolate_traceinv
* tests of traceinv
* rename the argument ``reorthogonalize`` on ``slq`` method to
  ``orthogonalize``. In ``hutchinson`` method, we have ``orthogonalize``
  argument, and it is better to make these two arguments have the same name,
  despite by orhtogonalize for the ``slq`` method, we mean re-orthogonalize.

* generate_matrix add analytic dense and sparse matrices
* generate_matrix add dtype argument

* doxygen for c_linear_operator and its derived classes

======
Issues
======

Template functions for the namespaces ``cublas_interface`` and
``cusparse_interface`` only work with gcc>7, which is a gcc bug. This is fine
with clang.

If ``this->device_buffer`` has to be set in sparse classes, the functions such
as ``dot()`` cannot be ``const``. So, either
    1. the constness should be removed from all base classes, or,
    2. somehow setting the buffer should be done outside of these functions,
       maybe in the constructors, or
    3. the namespaces be removed.


========================
Compile and Build Issues
========================

------------------
Local Installation
------------------

- Python 2.7:
  I dropped support for python 2.7, since
  ``scipy.special.cython_special.erfinv`` is not defined in the latest scipy
  that can be installed in python 2.7, which is scipy 1.2.3. The function
  ``erfinv`` exists in scipy as *python* function, but not as a *cyhton*
  function in ``cython_special``. The first version of scipy that includes
  ``erfinv`` as cython function is scipy 1.5.0.

- Pythn 3.5:
  For some reasons, this package cannot be installed on python 3.5. However,
  py35 is deprecated as of last year.

- CUDA support:
  Currently, it can be only enabled for linux and macos. CUDA support cannot be
  compiled in windows, since the funciton
  ``customize_windows_compiler_for_nvcc`` is not complete.

----
PyPi
----

- The CUDA installation on githib workflow is only available for linux and
  windows (using ``Jimver@cuda-toolkit``). This github action does not support
  macos. Also, my package cannot be compiled with CUDA on windows. Thus, the
  my package on pypi supports CUDA only on linux at the moment.

- For the linux build, I use ``Jimver@cuda-toolkit`` for ``build-linux.yaml``
  only, but not in ``deploy-pypi.yaml``. That is becase in pypi, we should
  build linux in ``manylinux`` docker image, and cuda should be installed
  inside the docker image. There is a script in ``.github/scripts`` that
  installs cuda 11-3 inside the CentOS linux of the ``manylinux2104`` image.

  Unfortunately, the size of manylinux wheel when this package is compiled
  with cuda is 407MB (without cuda, it is 8MB). The limit of upload size to
  pypi is 100MB, thus, the manylinux wheels cannot be uploaded to pypi at the
  moment. The problem is probabely the inclusion of cuda static libraries. One
  solution is to use ``--cudart shared`` in the linker arguments for nvcc. But
  I do not know how to add this to thee nvcc linker.

-----
Conda
-----

- For some reasosn, conda cannot build the package and this needs to be fixed.

=====
Ideas
=====

--------------------------------------------
``keep`` option for ``AffineMatrixFunction``
--------------------------------------------

For ``AffineMatrixFunction``, have an option to store all theta and tau to be
reused to next parameters. One way to do so is to bring the ``traceinv``
computation from the ``traceinv()`` function to be a member of
``LinearOperator`` class.

Here is how it should work:

1. On first run of `AffineMatrixFunction.traceinv()`` (or any other function
   such as ``logdet()``), all theta and tau are stored as member data of ``Aop``.
2. On the second call of the function (which the second function can be
   different than the previous function, as long as both used ``method='slq'``),
   the previous sample data (that and theta) are used. To case emerge:

   2.1. If within the existing samples, the results of the desired function
        converged within the given tolerance limit, no newer samples are needed.
        Thus, the function returns immediately.
   2.2. If the convergence has not been met, newer samples will be produced
        till the convergence is reached. The newer samples are also appended to
        the previous results.

.. code-block:: python

   >>> # keep argument lets the theta and tau to be stored with the cost of
   >>> # taking memory. Default is True.
   >>> Aop = AffineMatrixFunction(A, keep=True)

   >>> # The theta and tau are stored in Aop member data to be reused later
   >>> # Runtime: 10 seconds (just for example)
   >>> Aop.traceinv(method='slq', parameters=[1, 2], lanczos_degree=50,
                    min_num_samples=10, max_num_samples=100, error_rtol=1e-2)

   >>> # Here, we reuse the previous theta and tau
   >>> # Runtime: 0.0001 seconds
   >>> Aop.traceinv(method='slq', parameters=[3, 4], lanczos_degree=50,
                    min_num_samples=10, max_num_samples=100, error_rtol=1e-2)

   >>> # Because error_rtol is smaller, we might need to generate new samples
   >>> # and append to the previous samples
   >>> # Runtime: 5 seconds
   >>> Aop.traceinv(method='slq', parameters=[5, 6], lanczos_degree=50,
                    min_num_samples=10, max_num_samples=100, error_rtol=1e-3)

   >>> # Previous theta and tau from the previous results can be used for
   >>> # logdet or any other function, not just traceinv
   >>> # Runtime: 0.0001 seconds
   >>> Aop.logdet(method='slq', parameters=[7, 8], lanczos_degree=50,
                  min_num_samples=10, max_num_samples=100, error_rtol=1e-2)

   >>> # Here, all the previous theta and tau from previous samples are purged,
   >>> # since "lanczos_degree" is changed, which changes theta and tau sizes.
   >>> # Runtime: 10 seconds
    >>> Aop.traceinv(method='slq', parameters=[9, 10], lanczos_degree=60,
                     min_num_samples=10, max_num_samples=100, error_rtol=1e-3)

-----------------
Hutchinson Method
-----------------

Add convergence methods to the Hutchinson method, such as ``min_num_samples``,
``max_num_samples``, ``error_rtol``, ``error_atol``. Also add an option for
``reorthogonalization`` where the initial random vectors to be orthogonalizaed
(currently they are orthogonalized). Also an option for ``verbose`` to print
the results in a table just like the slq method, and an option for ``plot`` to
plot the convergence and samples.
