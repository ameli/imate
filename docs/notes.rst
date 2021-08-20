*****
Notes
*****

Some notes to myself when completing the documentation later.

* How to make the results of GPU and CPU identical for testing purposes:

  When num_gpu_devices and num_threads are the same, both cpu and gpu codes
  give identical results. This is due to the random number generator. For each
  thread, the random generator jumps the initial seed. Also, when an iteration
  on a single thread finishes, the next iteration continues with the next
  random number in the previous sequence in the memory of the generator.

  To make the result of both cpu and gpu exactly identical, do the followings:
  1. Set num_threads and num_gpu_devices to 1.
  2. In imate/_random_generator/split_mix_64.cpp, and in the constructor,
     initialize the seed with a fixed number, say 1234567890, not with time.
     This allows to run both the cpu code and gpu codes to start off by the
     same sequence of random numbers.

  If we set num_thread and num_gpu_devices to anything greater than one, the
  results might be different, since after each thread iteration, it is not
  guaranteed which thread continues the sequence of the previous random
  generator. But if min_num_samples is a large number, the results of both
  cpu and gpu should be very close.

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

* Anaconda wheels are build without dynamic loading (so the wheels bundle the
  cuda libraries). PyPi wheels are build with dynamic loading (so the wheels
  do not bundle with cuda libraries). The following is only related to the
  dynamic loading, meaning that only those wheels uploaded to PyPi.
  
  With dynamic loading (pypi wheels), the running machine should have the exact
  same CUDA version as the build machine. I thought this version match is only
  for the "major" cuda version. But apparently, the "minor" version should also
  match. For example, if the manylinux wheel is compiled on CUDA 10.2, it can
  only run on CUDA 10.2, but not 11, or even 10.1. I did not expect the later,
  since both 10.2 and 10.1 have the same SONAME (lib ABI), which is 10.

  Version 10.0 is quite different story, since it does not have the LT libs,
  such as libcublasLt.so. So, it make sense that it wheel build on 10.2 cannot
  run on 10.0.

  For 10.1, I get the some std::logic_error about some return string is NULL.
  I suspect this is one of the functions in `_cuda_dynamic_loading/*.cpp`.

* Anaconda wheels are build without dynamic loading, so they bundle with the
  cuda libraries. But, they still do not work fine. Here is how.

  An anaconda wheel that was built on cuda 10.2, can run fine on savio when
  cuda module is not loaded, or loaded. 

  An anaconda wheel that was built on cuda 11, can NOT run on savio when
  cuda module is not loaded, or loaded. I did not expect this, since it should
  run without needing to load any module in savio. Could this be the cuda
  "driver"? Because the device driver is not bundled with the wheel, and it
  should be installed with the machine. Cuda driver 11 is not available on
  savio.

====
Name
====

Implicit Matrix Trace Estimator: imte, >>"imate"<<, imtraes, "imtrest",
    "tracest", "imtest"
Fast Trace Estimator
"scikit-trace"

====
TODO
====

* Implement ``keep`` functionality for slq method.
* Other functions (besides traceinv and logdet)
* doxygen for c_linear_operator and its derived classes
* Get memory usage info for GPU. See for example:
  https://stackoverflow.com/questions/15966046/cudamemgetinfo-returns-same-amount-of-free-memory-on-both-devices-of-gtx-690

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

- pypy:
  Build on pypy is only suppported on Linux. The package cannot be built on
  pypy on windows and macos. On Linux, pypy-3.6 and pypy-3.7 is supported.

- CUDA support:
  CUDA is only availble in linux and windows. NVIDIA no longer supports CUDA in
  macos, and Apple does not include NVIDA in apple products either.

=====
Ideas
=====

---------
functions
---------

Encapsulate functions in a cdef class so that they can be passed from python to
slq method.

--------------------
Chebychev Hutchinson
--------------------

See trace estimation using Chebychev Hutchinson method:
https://nextjournal.com/akshayjain/traceEstimator02/

It can also be used to compute logdet:
https://nextjournal.com/akshayjain/logdet-via-chebyhutch


--------------------------------------------
``keep`` option for ``AffineMatrixFunction``
--------------------------------------------

For ``AffineMatrixFunction``, have an option to store all ``theta`` and ``tau``
to be reused to next parameters. One way to do so is to bring the ``traceinv``
computation from the ``traceinv()`` function to be a member of
``LinearOperator`` class.

Here is how it should work:

1. On the first run of `AffineMatrixFunction.traceinv()`` (or any other
   function such as ``logdet()``), all theta and tau are stored as member data
   of ``Aop``.
2. On the second call of the function (which the second function can be
   different than the previous function, as long as both of the calls used
   ``method='slq'``), the previous sample data (that and theta) are used. To
   case emerge:

   2.1. If within the existing samples, the results of the desired function
        converged within the given tolerance limit, no newer samples are needed.
        Thus, the function returns immediately.
   2.2. If the convergence has not been met, newer samples will be produced
        till the convergence is reached. The newer samples are also appended to
        the previous results.

.. code-block:: python

   >>> # keep argument allows the theta and tau to be stored with the cost of
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

   >>> # Because here the error_rtol is smaller, we might need to generate new
   >>> # samples, and append to the previous samples
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

==================
Method Limitations
==================

- Matrices where their eigenvalue spectra cannot be represented by a limited
  eigenvalues. If the lanczos degree is ``m``, and it the input matrix's
  eigenvalues have at most ``m`` significant eigenvalues, then the SLQ method
  performs well. Covariance matrices usually have such property, where most of
  their eigenvalues are zero zero, but a small number of them are significant.

=========================
Implementation Techniques
=========================

- Lazy evaluation in linear operator and copy data to gpu device.
- dynamic polymorphism to dispatch to linear operator derived classes.
- Static template to support float, double, and long double data types.
- Dynamic loading of CUDA libraries.
- Random generator for Rademacher distribution is implemented. This is near
  a hundred times faster than C's ``rand()`` function. The implementation uses
  xoshiro_265_star_star algorithm to generate 64-bit integers, which feeds to
  64 elements of array as +1 and -1 values. The initial seed uses split_mix
  random generator and itself is seeded by cpu time in microseconds.
  The random array generator can generate is thread-safe and can generate
  independent sequences of random numbers on each thread. The random array
  generator can be used on 2^64 parallel threads, each generating a sequence
  of 2^128 long.
- The basic algebra module seems to perform faster than OpenBlas. Not only
  that, for very large arrays, the dot product is more accurate than OpenBlas,
  since the reduction variable is cast to long double.


==================
Installation Notes
==================

--------
OpenBlas
--------

Install Openblas with conda. This is especially useful if you don't have admin access to install with apt.

.. code::

    conda install -c anaconda openblas

or with ``apt`` (needs admin access)

.. code::

    sudo apt-get install libopenblas-dev

Or in macos with brew by:

.. code::

    brew install openblas
