.. _install:

Install
*******

.. contents::

.. _install-wheels:

Install |project| From Wheels
=============================

Python wheels for |project| are available for various operating systems and Python versions on both PyPI and Anaconda Cloud.

Install with ``pip``
--------------------

|pypi|

Install |project| and its Python dependencies through `PyPI <https://pypi.org/project/imate>`_ by

.. prompt:: bash
    
    python -m pip install --upgrade pip
    python -m pip install imate

If you are using PyPy instead of Python, install with

.. prompt:: bash
    
    pypy -m pip install --upgrade pip
    pypy -m pip install imate

Install with ``conda``
----------------------

|conda-version|

Alternately, install |project| and its Python dependencies from `Anaconda Cloud <https://anaconda.org/s-ameli/imate>`_ by

.. prompt:: bash

    conda install -c s-ameli imate -y

.. _virtual-env:

Install |project| in Virtual Environments
=========================================

If you do not want the installation to occupy your main python's site-packages (either you are testing or the dependencies may clutter your existing installed packages), install the package in an isolated virtual environment. Two common virtual environments are :ref:`virtualenv <virtualenv_env>` and :ref:`conda <conda_env>`.

.. _virtualenv_env:

Install in ``virtualenv`` Environment
-------------------------------------

1. Install ``virtualenv``:

   .. prompt:: bash

       python -m pip install virtualenv

2. Create a virtual environment and give it a name, such as ``imate_env``

   .. prompt:: bash

       python -m virtualenv imate_env

3. Activate python in the new environment

   .. prompt:: bash

       source imate_env/bin/activate

4. Install ``imate`` package with any of the :ref:`above methods <install-wheels>`. For instance:

   .. prompt:: bash

       python -m pip install imate
   
   Then, use the package in this environment.

5. To exit from the environment

   .. prompt:: bash

       deactivate

.. _conda_env:

Install in ``conda`` Environment
--------------------------------

In the followings, it is assumed `anaconda <https://www.anaconda.com/products/individual#Downloads>`_ (or `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_) is installed.

1. Initialize conda

   .. prompt:: bash

       conda init

   You may need to close and reopen your terminal after the above command. Alternatively, instead of the above, you can do

   .. prompt:: bash

       sudo sh $(conda info --root)/etc/profile.d/conda.sh

2. Create a virtual environment and give it a name, such as ``imate_env``

   .. prompt:: bash

       conda create --name imate_env -y

   The command ``conda info --envs`` shows the list of all environments. The current environment is marked by an asterisk in the list, which should be the default environment at this stage. In the next step, we will change the current environment to the one we created.

3. Activate the new environment

   .. prompt:: bash

       source activate imate_env

4. Install ``imate`` with any of the :ref:`above methods <install-wheels>`. For instance:

   .. prompt:: bash

       conda install -c s-ameli imate
   
   Then, use the package in this environment.

5. To exit from the environment

   .. prompt:: bash

       conda deactivate

.. _compile-imate:

Optional Runtime Dependencies
=============================

Runtime libraries are not required to be present during the installation of |project|. However, they may be required to be installed during running |project|.

CUDA Toolkit and NVIDIA Graphic Driver (`Optional`)
---------------------------------------------------

To use GPU devices, install NVIDIA Graphic Driver and CUDA Toolkit. See the instructions below.

* :ref:`Install NVIDIA Graphic Driver <install-graphic-driver>`.
* :ref:`Install CUDA Toolkit <install-cuda-toolkit>`.

Sparse Suite (`Optional`)
-------------------------

`Suite Sarse <https://people.engr.tamu.edu/davis/suitesparse.html>`_ is a library for efficient calculations on sparse matrices. |project| does not require this library as it has its own library for sparse matrices. However, if this library is available, |project| uses it.

.. note::

    The Sparse Suite library is only used for those functions in |project| that uses the Cholesky decomposition method by passing ``method=cholesky`` argument to the functions. See :ref:`API reference for Functions <Functions>` for details. 

1. Install Sparse Suite development library by

   .. tab-set::

       .. tab-item:: Ubuntu/Debian
          :sync: ubuntu

          .. prompt:: bash

              sudo apt install libsuitesparse-dev

       .. tab-item:: CentOS 7
          :sync: centos

          .. prompt:: bash

              sudo yum install libsuitesparse-devel

       .. tab-item:: RHEL 9
          :sync: rhel

          .. prompt:: bash

              sudo dnf install libsuitesparse-devel

       .. tab-item:: macOS
          :sync: osx

          .. prompt:: bash

              sudo brew install suite-sparse

   Alternatively, if you are using *Anaconda* python distribution (on either of the operating systems), install Suite Sparse by:

   .. prompt:: bash

       sudo conda install -c conda-forge suitesparse

2. Install ``scikit-sparse`` python package:

   .. prompt:: bash
       
       python -m pip install scikit-sparse

OpenBLAS (`Optional`)
---------------------

`OpenBLAS <https://www.openblas.net/>`_ is a library for efficient dense matrix operations. |project| does not require this library as it has its own library for dense matrices. However, if you compiled |project| to use OpenBLAS (see :ref:`Compile from Source <compile-imate>`), OpenBLAS library should be available at runtime.

.. note::

    A default installation of |project| through ``pip`` or ``conda`` does not use OpenBLAS, and you may skip this section.

Install OpenBLAS library by

.. tab-set::

   .. tab-item:: Ubuntu/Debian
      :sync: ubuntu

      .. prompt:: bash

            sudo apt install libopenblas-dev

   .. tab-item:: CentOS 7
      :sync: centos

      .. prompt:: bash

          sudo yum install openblas-devel

   .. tab-item:: RHEL 9
      :sync: rhel

      .. prompt:: bash

          sudo dnf install openblas-devel

   .. tab-item:: macOS
      :sync: osx

      .. prompt:: bash

          sudo brew install openblas

Alternatively, you can install OpenBLAS using ``conda``:

.. prompt:: bash

    conda install -c anaconda openblas

.. _compile-source:

Compile from Source
===================

When to Compile |project|
-------------------------

Generally, it is not required to compile |project| as the installation through ``pip`` and ``conda`` contains most of its features, including support for GPU devices. You may compile |project| if you want to:

* modify |project|.
* use `OpenBLAS` instead of the built-in matrix library of |project|.
* build |project| for a `specific version` of CUDA Toolkit.
* disable the `dynamic loading` feature of |project| for CUDA libraries.
* enable `debugging mode`.
* or, build this `documentation`.

Otherwise, install |project| through the :ref:`Python Wheels <install-wheels>`.

This section walks you through the compilation process.

Install C++ Compiler (`Required`)
---------------------------------

Compile |project| with either of GCC, Clang/LLVM, or Intel C++ compiler on UNIX operating systems. For Windows, compile |project| with `Microsoft Visual Studio (MSVC) Compiler for C++ <https://code.visualstudio.com/docs/cpp/config-msvc#:~:text=You%20can%20install%20the%20C,the%20C%2B%2B%20workload%20is%20checked.>`_.

.. rubric:: Install GNU GCC Compiler


.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            sudo apt install build-essential

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum group install "Development Tools"

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf group install "Development Tools"

    .. tab-item:: macOS
        :sync: osx

        .. prompt:: bash

            sudo brew install gcc libomp

Then, export ``C`` and ``CXX`` variables by

.. prompt:: bash

  export CC=/usr/local/bin/gcc
  export CXX=/usr/local/bin/g++

.. rubric:: Install Clang/LLVN Compiler
  
.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            sudo apt install clang

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum install yum-utils
            sudo yum-config-manager --enable extras
            sudo yum makecache
            sudo yum install clang

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf install yum-utils
            sudo dnf config-manager --enable extras
            sudo dnf makecache
            sudo dnf install clang

    .. tab-item:: macOS
        :sync: osx

        .. prompt:: bash

            sudo brew install llvm libomp-dev

Then, export ``C`` and ``CXX`` variables by

.. prompt:: bash

  export CC=/usr/local/bin/clang
  export CXX=/usr/local/bin/clang++

.. rubric:: Install Intel oneAPI Compiler

To install `Intel Compiler` see `Intel oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=aptpackagemanager>`_.

Install OpenMP (`Required`)
---------------------------

OpenMP comes with the C++ compiler installed. However, you may alternatively install it directly on UNIX. Install `OpenMP` library on UNIX as follows:

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            sudo apt install libgomp1 -y

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum install libgomp -y

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf install libgomp -y

    .. tab-item:: macOS
        :sync: osx

        .. prompt:: bash

            sudo brew install libomp

.. note::

    In *macOS*, starting from ``libomp`` with version ``15`` and above, Homebrew installs OpenMP as *keg-only*. As a result, create the following symbolic links to be able to use the OpenMP installation

    .. prompt:: bash

        ln -s /usr/local/opt/libomp/include/omp-tools.h /usr/local/include/omp-tools.h
        ln -s /usr/local/opt/libomp/include/omp.h /usr/local/include/omp.h
        ln -s /usr/local/opt/libomp/include/ompt.h /usr/local/include/ompt.h
        ln -s /usr/local/opt/libomp/lib/libomp.a /usr/local/lib/libomp.a
        ln -s /usr/local/opt/libomp/lib/libomp.dylib /usr/local/lib/libomp.dylib

.. _install-openblas:

OpenBLAS (`Optional`)
---------------------

|project| can be compiled with and without OpenBLAS. If you are compiling |project| with OpenBLAS, install OpenBLAS library by

.. tab-set::

   .. tab-item:: Ubuntu/Debian
      :sync: ubuntu

      .. prompt:: bash

            sudo apt install libopenblas-dev

   .. tab-item:: CentOS 7
      :sync: centos

      .. prompt:: bash

          sudo yum install openblas-devel

   .. tab-item:: RHEL 9
      :sync: rhel

      .. prompt:: bash

          sudo dnf install openblas-devel

   .. tab-item:: macOS
      :sync: osx

      .. prompt:: bash

          sudo brew install openblas

Alternatively, you can install OpenBLAS using ``conda``:

.. prompt:: bash

    conda install -c anaconda openblas

.. note::

    To build |project| with OpenBLAS, you should also set ``USE_CBLAS`` environment variable as described in :ref:`Configure Compile-Time Environment Variables <config-env-variables>`.

.. _install-cuda:

Install CUDA Compiler (`Optional`)
----------------------------------

To use |project| on GPU devices, it should be compiled with CUDA compiler. Skip this part if you are not using GPU.

.. note::

    The minimum version of CUDA to compile |project| is `CUDA 10.0`.

.. attention::

    NVIDIA does not support macOS. You can install the NVIDIA CUDA Toolkit on Linux and Windows only.


It is not required to install the entire CUDA Toolkit. Install only the CUDA compiler and the development libraries of cuBLAS and cuSparse by

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            sudo apt install -y \
                cuda-nvcc-11-7 \
                libcublas-11-7 \
                libcublas-dev-11-7 \
                libcusparse-11-7 -y \
                libcusparse-dev-11-7

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum install --setopt=obsoletes=0 -y \
                cuda-nvcc-11-7.x86_64 \
                cuda-cudart-devel-11-7.x86_64 \
                libcublas-11-7.x86_64 \
                libcublas-devel-11-7.x86_64 \
                libcusparse-11-7.x86_64 \
                libcusparse-devel-11-7.x86_64

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf install --setopt=obsoletes=0 -y \
                cuda-nvcc-11-7.x86_64 \
                cuda-cudart-devel-11-7.x86_64 \
                libcublas-11-7.x86_64 \
                libcublas-devel-11-7.x86_64 \
                libcusparse-11-7.x86_64 \
                libcusparse-devel-11-7.x86_64

Update ``PATH`` with the CUDA installation location by

.. prompt:: bash

    echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
    source ~/.bashrc

Check if the CUDA compiler is available with ``which nvcc``.

.. note::

    To build |project| with CUDA, you should also set ``CUDA_HOME``, ``USE_CUDA``, and optionally set ``CUDA_DYNAMIC_LOADING`` environment variables as described in :ref:`Configure Compile-Time Environment Variables <config-env-variables>`.

Load CUDA Compiler on GPU Cluster (`Optional`)
----------------------------------------------

This section is relevant if you are using GPU on a cluster and skip this section otherwise.

On a GPU cluster, chances are the CUDA Toolkit is already installed. If the cluster uses the `module` interface, load CUDA as follows.

First, check if a CUDA module is available by

.. prompt:: bash

    module avail

Load both CUDA and GCC by

.. prompt:: bash

    module load cuda gcc

You may specify CUDA version if multiple CUDA versions are available, such as by

.. prompt:: bash

    module load cuda/11.7 gcc/6.3

You may check if CUDA Compiler is available with ``which nvcc``.

.. _config-env-variables:

Configure Compile-Time Environment Variables (`Optional`)
---------------------------------------------------------

Set the following environment variables as desired to configure the compilation process.

.. glossary::

    ``CUDA_HOME``, ``CUDA_PATH``, ``CUDA_ROOT``

        These variables are relevant only if you are compiling with the CUDA compiler. :ref:`Install CUDA Toolkit <install-cuda>` and specify the home directory of CUDA Toolkit by setting either of these variables. The home directory should be a path containing the executable ``/bin/nvcc`` (or ``\bin\nvcc.exe`` on Windows). For instance, if ``/usr/local/cuda/bin/nvcc`` exists, export the following:

        .. tab-set::

            .. tab-item:: UNIX
                :sync: unix

                .. prompt:: bash

                    export CUDA_HOME=/usr/local/cuda

            .. tab-item:: Windows (Powershell)
                :sync: win

                .. prompt:: powershell

                    $env:export CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7"

    ``USE_CUDA``

        This variable is relevant only if you are compiling with the CUDA compiler. By default, this variable is set to `0`. To compile |project| with CUDA, :ref:`install CUDA Toolkit <install-cuda>` and set this variable to `1` by

        .. tab-set::

            .. tab-item:: UNIX
                :sync: unix

                .. prompt:: bash

                    export USE_CUDA=1

            .. tab-item:: Windows (Powershell)
                :sync: win

                .. prompt:: powershell

                    $env:export USE_CUDA = "1"

    ``CUDA_DYNAMIC_LOADING``

        This variable is relevant only if you are compiling with the CUDA compiler. By default, this variable is set to `0`.  When |project| is complied with CUDA, the CUDA runtime libraries bundle with the final installation of |project| package, making it over 700MB. While this is generally not an issue for most users, often a small package is preferable if the installed package has to be distributed to other machines. To this end, enable the custom-made `dynamic loading` feature of |project|. In this case, the CUDA libraries will not bundle with the |project| installation, rather, |project| is instructed to load the existing CUDA libraries of the host machine at runtime. To enable dynamic loading, make sure :ref:`CUDA Toolkit <install-cuda>` is installed, then set this variable to `1` by

        .. tab-set::

            .. tab-item:: UNIX
                :sync: unix

                .. prompt:: bash

                    export CUDA_DYNAMIC_LOADING=1

            .. tab-item:: Windows (Powershell)
                :sync: win

                .. prompt:: powershell

                    $env:export CUDA_DYNAMIC_LOADING = "1"

    ``CYTHON_BUILD_IN_SOURCE``

        By default, this variable is set to `0`, in which the compilation process generates source files outside of the source directory, in ``/build`` directry. When it is set to `1`, the build files are generated in the source directory. To set this variable, run

        .. tab-set::

            .. tab-item:: UNIX
                :sync: unix

                .. prompt:: bash

                    export CYTHON_BUILD_IN_SOURCE=1

            .. tab-item:: Windows (Powershell)
                :sync: win

                .. prompt:: powershell

                    $env:export CYTHON_BUILD_IN_SOURCE = "1"

        .. hint::

            If you generated the source files inside the source directory by setting this variable, and later you wanted to clean them, see :ref:`Clean Compilation Files <clean-files>`.

    ``CYTHON_BUILD_FOR_DOC``

        Set this variable if you are building this documentation. By default, this variable is set to `0`. When it is set to `1`, the package will be built suitable for generating the documentation. To set this variable, run

        .. tab-set::

            .. tab-item:: UNIX
                :sync: unix

                .. prompt:: bash

                    export CYTHON_BUILD_FOR_DOC=1

            .. tab-item:: Windows (Powershell)
                :sync: win

                .. prompt:: powershell

                    $env:export CYTHON_BUILD_FOR_DOC = "1"

        .. warning::

            Do not use this option to build the package for `production` (release) as it has a slower performance. Building the package by enabling this variable is only suitable for generating the documentation.

        .. hint::

            By enabling this variable, the build will be `in-source`, similar to setting ``CYTHON_BUILD_IN_SOURCE=1``. To clean the source directory from the generated files, see :ref:`Clean Compilation Files <clean-files>`.

    ``USE_CBLAS``

        By default, this variable is set to `0`. Set this variable to `1` if you want to use OpenBLAS instead of the built-in library of |project|. :ref:`Install OpenBLAS <install-openblas>` and set

        .. tab-set::

            .. tab-item:: UNIX
                :sync: unix

                .. prompt:: bash

                    export USE_CBLAS=1

            .. tab-item:: Windows (Powershell)
                :sync: win

                .. prompt:: powershell

                    $env:export USE_CBLAS = "1"

    ``DEBUG_MODE``

        By default, this variable is set to `0`, meaning that |project| is compiled without debugging mode enabled. By enabling debug mode, you can debug the code with tools such as ``gdb``. Set this variable to `1` to enable debugging mode by

        .. tab-set::

            .. tab-item:: UNIX
                :sync: unix

                .. prompt:: bash

                    export DEBUG_MODE=1

            .. tab-item:: Windows (Powershell)
                :sync: win

                .. prompt:: powershell

                    $env:export DEBUG_MODE = "1"

        .. attention::

            With the debugging mode enabled, the size of the package will be larger and its performance may be slower, which is not suitable for `production`.

Compile and Install
-------------------

|repo-size|

Get the source code of |project| from the GitHub repository by

.. prompt:: bash

    git clone https://github.com/ameli/imate.git
    cd imate

To compile and install, run

.. prompt:: bash

    python setup.py install

The above command may need ``sudo`` privilege. 

.. rubric:: A Note on Using ``sudo``

If you are using ``sudo`` for the above command, add ``-E`` option to ``sudo`` to make sure the environment variables (if you have set any) are accessible to the root user. For instance

.. tab-set::

    .. tab-item:: UNIX
        :sync: unix

        .. code-block:: Bash
            :emphasize-lines: 5

            export CUDA_HOME=/usr/local/cuda
            export USE_CUDA=1
            export CUDA_DYNAMIC_LOADING=1

            sudo -E python setup.py install

    .. tab-item:: Windows (Powershell)
        :sync: win

        .. code-block:: PowerShell
            :emphasize-lines: 5

            $env:export CUDA_HOME = "/usr/local/cuda"
            $env:export USE_CUDA = "1"
            $env:export CUDA_DYNAMIC_LOADING = "1"

            sudo -E python setup.py install

Once the installation is completed, check the package can be loaded by

.. prompt:: bash

    cd ..  # do not load imate in the same directory of the source code
    python -c "import imate; imate.info()"

The output to the above command should be similar to the following:

.. code-block:: text

    imate version   : 0.15.0
    processor       : Intel(R) Xeon(R) CPU E5-2623 v3 @ 3.00GHz
    num threads     : 8
    gpu device      : GeForce GTX 1080 Ti
    num gpu devices : 4
    cuda version    : 11.2.0
    process memory  : 61.4 (Mb)

.. attention::

    Do not load |project| if your current working directory is the root directory of the source code of |project|, since python cannot load the installed package properly. Always change the current directory to somewhere else (for example, ``cd ..`` as shown in the above).

.. _clean-files:
   
.. rubric:: Cleaning Compilation Files

If you set ``CYTHON_BUILD_IN_SOURCE`` or ``CYTHON_BUILD_FOR_DOC`` to ``1``, the output files of Cython's compiler will be generated inside the source code directories. To clean the source code from these files (`optional`), run the following:

.. prompt:: bash

    python setup.py clean

Generate Documentation
======================

The documentation consists of a Doxygen documentation and a Sphinx documentation. You should generate the Doxygen documentation first. Before generating the Sphinx documentation, you should compile the package.

Get the source code from the GitHub repository.

.. prompt:: bash

    git clone https://github.com/ameli/imate.git
    cd imate

If you already had the source code, clean it from any previous build (especially if you built `in-source`):

.. prompt:: bash

    python setup.py clean

Generate Doxygen Documentation
------------------------------

Install `doxygen` and `graphviz` by

.. tab-set::

   .. tab-item:: Ubuntu/Debian
      :sync: ubuntu

      .. prompt:: bash

            sudo apt install doxygen graphviz -y

   .. tab-item:: CentOS 7
      :sync: centos

      .. prompt:: bash

          sudo yum install doxygen graphviz -y

   .. tab-item:: RHEL 9
      :sync: rhel

      .. prompt:: bash

          sudo dnf install doxygen graphviz -y

   .. tab-item:: macOS
      :sync: osx

      .. prompt:: bash

          sudo brew install doxygen graphviz -y

   .. tab-item:: Windows (Powershell)
      :sync: win

      .. prompt:: powershell

          scoop install doxygen graphviz

.. attention::

    Make sure you ran ``python setup.py clean`` as mentioned previously. Otherwise, if the source directory is not cleaned from any previous built (if there is any), Doxygen unwantedly generates all auto-generated cython files.

Generating the Doxygen documentation by

.. prompt:: bash

    cd docs/doxygen
    doxygen doxyfile.in
    cd ../..

Compile Package
---------------

Set ``CYTHON_BUILD_FOR_DOC`` to `1` (see :ref:`Configure Compile-Time Environment variables <config-env-variables>`). Compile and install the package by

.. tab-set::

    .. tab-item:: UNIX
        :sync: unix

        .. prompt:: bash

            export CYTHON_BUILD_FOR_DOC=1
            export USE_CUDA=0
            sudo -E python setup.py install

    .. tab-item:: Windows (Powershell)
        :sync: win

        .. prompt:: powershell

            $env:export CYTHON_BUILD_FOR_DOC = "1"
            $env:export USE_CUDA = "0"
            sudo -E python setup.py install

Generate Sphinx Documentation
-----------------------------

Install `Pandoc <https://pandoc.org/>`_ by

.. tab-set::

   .. tab-item:: Ubuntu/Debian
      :sync: ubuntu

      .. prompt:: bash

            sudo apt install pandoc -y

   .. tab-item:: CentOS 7
      :sync: centos

      .. prompt:: bash

          sudo yum install pandoc -y

   .. tab-item:: RHEL 9
      :sync: rhel

      .. prompt:: bash

          sudo dnf install pandoc -y

   .. tab-item:: macOS
      :sync: osx

      .. prompt:: bash

          sudo brew install pandoc -y

   .. tab-item:: Windows (Powershell)
      :sync: win

      .. prompt:: powershell

          scoop install pandoc

Install the requirements for the Sphinx documentation by

.. prompt:: bash

    python -m pip install -r docs/requirements.txt

The above command installs the required packages in Python's path directory. Make sure python's directory is on the `PATH`, for instance, by

.. tab-set::

    .. tab-item:: UNIX
        :sync: unix

        .. prompt:: bash

            PYTHON_PATH=`python -c "import os, sys; print(os.path.dirname(sys.executable))"`
            export PATH=${PYTHON_PATH}:$PATH

    .. tab-item:: Windows (Powershell)
        :sync: win

        .. prompt:: powershell

            $PYTHON_PATH = (python -c "import os, sys; print(os.path.dirname(sys.executable))")
            $env:Path += ";$PYTHON_PATH"

Now, build the documentation:

.. tab-set::

    .. tab-item:: UNIX
        :sync: unix

        .. prompt:: bash

            make clean html --directory=docs

    .. tab-item:: Windows (Powershell)
        :sync: win

        .. prompt:: powershell

            cd docs
            make.bat clean html

The main page of the documentation can be found in ``/docs/build/html/index.html``. 

Test with ``pytest``
====================

|codecov-devel|

The package can be tested by running several `test scripts <https://github.com/ameli/imate/tree/main/tests>`_, which test all `sub-packages <https://github.com/ameli/imate/tree/main/imate>`_ and `examples <https://github.com/ameli/imate/tree/main/examples>`_.

Clone the source code from the repository and install the required test packages by

.. prompt:: bash

    git clone https://github.com/ameli/imate.git
    cd imate
    python -m pip install -r tests/requirements.txt
    python setup.py install

To automatically run all tests, use ``pytest`` which is installed by the above commands.

.. prompt:: bash

    mv imate imate-do-not-import
    pytest

.. attention::

    To properly run ``pytest``, rename ``/imate/imate`` directory as shown in the above code. This makes ``pytest`` to properly import |project| from the installed location, not from the source code directory.

.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/imate
   :target: https://codecov.io/gh/ameli/imate
.. |implementation| image:: https://img.shields.io/pypi/implementation/imate
.. |pyversions| image:: https://img.shields.io/pypi/pyversions/imate
.. |format| image:: https://img.shields.io/pypi/format/imate
.. |pypi| image:: https://img.shields.io/pypi/v/imate
.. |conda| image:: https://anaconda.org/s-ameli/imate/badges/installer/conda.svg
   :target: https://anaconda.org/s-ameli/imate
.. |platforms| image:: https://img.shields.io/conda/pn/s-ameli/imate?color=orange?label=platforms
   :target: https://anaconda.org/s-ameli/imate
.. |conda-version| image:: https://img.shields.io/conda/v/s-ameli/imate
   :target: https://anaconda.org/s-ameli/imate
.. |release| image:: https://img.shields.io/github/v/tag/ameli/imate
   :target: https://github.com/ameli/imate/releases/
.. |conda-platform| image:: https://anaconda.org/s-ameli/imate/badges/platforms.svg
   :target: https://anaconda.org/s-ameli/imate
.. |repo-size| image:: https://img.shields.io/github/repo-size/ameli/imate
   :target: https://github.com/ameli/imate
