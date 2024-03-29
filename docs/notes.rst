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
  ALL of its base classes, and even the other non-CUDA classes that are
  derived from the ``cLinearOperator`` class has to be non ``const`` member
  functions.

  If I can be certain that the buffer size is always zero, I can return back
  the const-ness to these functions and do not call the allocate buffer.
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

* ``scikit-sparse`` package computes the Cholesky decomposition of the mixed
  matrix :math:`$\mathbf{A} + \beta \mathbf{I}$`, which effectively can be used
  to compute its traceinv. See:
  https://scikit-sparse.readthedocs.io/en/latest/cholmod.html#sksparse.cholmod.cholesky
  However, this is the full-rank update on the Cholesky factorization of the
  matrix :math:`\mathbf{A}`, which is O(n^3) expensive. See:
  https://scicomp.stackexchange.com/questions/10630/full-rank-update-to-cholesky-decomposition

====
TODO
====

* Implement ``keep`` functionality for slq method.
* Hutchinson method can be implemented in C++ and also in CUDA on GPU.
* Other functions (besides traceinv and logdet)
* doxygen for c_linear_operator and its derived classes
* Get memory usage info for GPU. See for example:
  https://stackoverflow.com/questions/15966046/cudamemgetinfo-returns-same-amount-of-free-memory-on-both-devices-of-gtx-690
* for the return of functions, instead of outputting (trace, info) tuple, only
  return trace. However, in the arguments, include "full_output=False". If
  True, it then outputs the dictionary of info. See scipy.optimize.fsolve.
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html
* Check compilation with CUDA 12.
* Imate compiled with scipy==1.9.3 leads to the following runtime error:

    File "imate/_c_trace_estimator/py_c_trace_estimator.pyx", line 1, in init
    imate._c_trace_estimator.py_c_trace_estimator
    ImportError: scipy.special.cython_special does not export expected C
    function __pyx_fuse_0erfinv

  When imate is installed with conda, unfortunately, scipy=1.9.3 is installed
  with imate, and it causes the above error. This also causes some tests to
  fail (test/test_logdet.py, etc) when running deploy_conda.yaml.

  The issue seems to be related to inverse of error function. An alternative
  method (instead of erfinv from scipy.special) is to implement erfinv myself.
  See:

  Some short codes in C for inverse off error functions
  https://stackoverflow.com/questions/27229371/inverse-error-function-in-c

  or

  A github script in C++
  https://github.com/lakshayg/erfinv

  I may do something like:

  try:
    from scipy.special.cython_special cimport erfinv
  except:
    from ._erfinv cimport erfinv  # my implementation

========================
Compile and Build Issues
========================

-----------------------------------------
Compile issues arising specifically on CI
-----------------------------------------

The configurations below raise issue, not because these configurations cannot
be compiled, rather, they arise on continuous integration (CI) environments.

.....
Linux
.....

- pypy on linux AARCH64:
  We build all AARCH64 wheels on Cirrus CI. For CPython, we build wheels on
  both deploy-conda and deploy-pypi. For PyPy, we only build wheels on
  deploy-pypi (we do not build pypy wheels on conda).

  The PyPy wheels for linux (which are built with cuda support) on aarch64
  on cirrus ci takes more than 60 minutes, and cirrus ci terminates these
  jobs. As such, we do not support wheels for pypy-linux-aarch64.

  In contrast, pypy-macos-arm64 builds fine on cirrus ci, and this is because
  on macos, we do not support cuda, hence, the compile time is not long.

-----------------------
Issues with local build
-----------------------

.....
MacOS
.....

We do not support CUDA for macos, as Apply do not support NVIDIA GPUs.

.......
Windows
.......

The following compilation issues are not due to CI runners, rather these
configurations below cannot be compiled even on a local machine.

- pypy on windows:
  Build on pypy is only supported on Linux and macos. The package cannot be
  built on pypy on windows. This is because imate depends on scipy not only at
  runtime, but also "at compile time" to compile lapack dependency (recall
  that imate "cimports" scipy.linalg.cython_lapack). Whenever we "import
  (not import, but cimport), that dependency becomes compile-time dependency.

  Scipy does not have wheel on windows for PyPy. Hence, PyPy tries to
  compile scipy from source whenever PyPy compiles imate. But there are two
  issues with PyPy compiling scipy:
  1. The compilation process raises error that gfortran is not found.
     This can be easily resolved by: "choco install mingw". As such, the
     meson.build (build manager in scipy) will use mingw rather than MSVC, and
     mingw has fortran.
  2. After the above issue is resolved, another error arises, ans that is that
     build process cannot find openblas. Scipy finds openblas by internally
     installing a package called 'scipy_openblas32'. But even installing it
     does not fix the issue. As of now, I cannot figure how to resolve this.

  Because of the issue, we do not support wheelsL
  - pp*-win-arm64 and
  - pp*-win-amd64.

- ARM64 on windows:
  I built all ARM64 wheels for Linux and MacOS on native ARM64 machines on
  Cirrus CI. However, for Windows, the ARM64 wheels can be cross-compiled on
  an X86_64 machine (not native build is needed). This can be done on github
  runners which are X86_64 machines, and passing ARM64 flags to cibuildwheel.

  Cibuildwheel can cross-compile for ARM64 (from a X86_64 host machine) for
  windows only if the python package is built based on setuptools, but not
  based on meson.build. As such, my other package, special_functions, which
  uses meson.build, cannot be build for Windows ARM64. However, glearn can, and
  indeed, it compiles for win-arm64 just fine, as glearn uses setuptools.

  Imate also uses setuptools, however, it raises some errors when it is cross
  compiled for win-arm64. I think this is due to CUDA. So, disabling cuda might
  allow building for arm64.

  Even if I can build imate for win-arm64, it is still useless, as at runtime,
  imate needs numpy and scipy. But numpy and scipy do not provide win-arm64
  wheels neither in cpython nor in pypy. Thus, a user of the package will not
  have these essential packages (numpy, scipy), even if I provide them wheel
  for imate.

  Because of the above issue, we do not support wheels:
  - pp*-win-arm64, and
  - cp*-win-arm64.

  The only windows-based wheel that we support is:
  - cp*-win-amd64 (X86_64 and CPyhton on windows)

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
  their eigenvalues are zero, but a small number of them are significant.

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
- The basic algebra module seems to perform faster than OpenBLAS. Not only
  that, for very large arrays, the dot product is more accurate than OpenBLAS,
  since the reduction variable is cast to long double.

=============
Documentation
=============

Things yet remained in the documentation to be completed:

* docs/source/performance/interpolation.rst
* a Few more tutorials in jupyter notebook
* Incorporate /imate/examples (reproduce results of  interpolation paper) into
  the documentation.
