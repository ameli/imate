*******
Install
*******

===================
Python Dependencies
===================

The python dependency packages are:

* **Required:** ``numpy`` and ``scipy``.
* **Required:** ``matplotlib`` and ``seaborn``, but only required to run the :ref:`Examples`.
* **Optional:** ``ray`` and ``scikit-sparse`` can improve performance, but not required.

.. note::
    By installing TraceInv :ref:`below <Install TraceInv>`, the *required* dependencies (but not the *optional* packages) will be installed automatically and no other action is needed. However, if desired, the *optional* packages should be installed :ref:`manually <Install Optional Packages>`.

================
Install TraceInv
================

TraceInv can be installed on Linux, macOS, and Windows platforms and supports both python 2 and 3. Install by either of the following ways:

* **Method 1: Through PyPi package.** The recommended way to install TraceInv and its dependencies is through the package available at `PyPi <https://pypi.org/project/TraceInv>`_ by

  ::
      
      python -m pip install TraceInv

* **Method 2: Through Conda package.** Install TraceInv and its dependencies through the package available at `Conda <https://anaconda.org/conda-forge/TraceInv>`_ by

  ::

      conda install -c conda-forge TraceInv

* **Method 3: Build source locally.**
  Clone the source code, build locally, and install by
  
  ::

      git clone https://github.com/ameli/TraceInv.git
      cd TraceInv
      python setup build
      python setup install

  Note that the last line in the above many need to be run with ``sudo``.

  The third installation method does not install the dependencies automatically and they should be installed separately such as with ``pip`` below. In the same root directory of the package (where the file ``requirements.txt`` can be found) run

  ::

      python -m pip install -r requirements.txt

  Alternatively, the dependencies can be installed with ``conda`` by

  ::

      conda install --file requirements.txt

=======================================
Install TraceInv in Virtual Environment
=======================================

If you do not want the :ref:`above <Install TraceInv>` installation occupy your python site packages (either you are testing or the dependencies mess with your existing installed packages), you may install and try TraceInv in a virtual environment.

1. Install ``virtualenv``:

   ::

       python -m pip intall virtualenv

2. Create a virtual environment called ``TraceInvEnv``

   ::

       python -m virtualenv TraceInvEnv

3. Activate python in the new environment

   ::

       source TraceInvEnv/bin/activate

   Now, the python and pip is sourced from the new environment.

4. Install TraceInv with any of the :ref:`above <Install TraceInv>` methods and try it this python environment.

5. To exit from the environment

   ::

       deactivate

=========================
Install Optional Packages
=========================

Installing the optional packages below can improve the performance for some of the functionalities, but not necessary. 

.. _InstallRay:

---------------
Install ``ray``
---------------

::

    python -m pip install ray

When ``ray`` is needed:
    To run the :ref:`examples <Examples>`, you may install the ``ray`` package to leverage the parallel processing used to generate large sparse matrices. However, the examples produce results without installing ``ray``.


.. _InstallScikitSparse:

-------------------------
Install ``scikit-sparse``
-------------------------

1. Install `Suite Sarse <https://people.engr.tamu.edu/davis/suitesparse.html>`_ development library ``libsuitesparse-dev`` using ``apt`` package manager in Debian-based Linux distros (such as *Debian, Ubuntu, Mint*) by
   
   ::

       sudo apt install libsuitesparse-dev  

   Replace ``apt`` in the above with the native package manager of your operating system, such as ``yum`` for  *Redhat, Fedora, and CentOS Linux*, ``pacman`` for *Arch Linux*, and ``brew`` for *macOS*.

   Alternatively, if you are using *Anaconda* python distribution (on either of the operating systems), install Suite Sparse by:

   ::

       sudo conda install -c conda-forge suitesparse

2. Install ``scikit-sparse`` python package:

   ::
       
       python -m pip install scikit-sparse

When ``scikit-sparse`` is needed:
    In ``TraceInv`` package, one of the methods to compute the trace of a matrix is by the *Cholesky decomposition*. If the input matrix is *sparse*, the Cholesky decomposition is computed using ``scikit-sparse``. But if this package is not installed, the ``scipy`` package is used instead.
