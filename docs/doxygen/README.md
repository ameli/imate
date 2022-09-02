## How to Build


Compile from Source
===================

When to Compile |project|
-------------------------

Generally, it is not required to compile |project| as the installation through ``pip`` and ``conda`` contains most of its features, including support for GPU devices. You may compile |project| if you want to:

* modify |project|.
* use `OpenBLAS` instead of the built-in matrix library of |project|.
* build |project| for a `specific version` of CUDA Toolkit.
* disable `dynamic loading` feature of |project| for CUDA libraries.
* enable `debugging mode`.
* or, build this `documentation`.

Otherwise, install |project| through the :ref:`Python Wheels <install-wheels>`.

This section walks you through the compilation process.

Install C++ Compiler and OpenMP (`Required`)
--------------------------------------------

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

OpenMP comes with the C++ compiler installed in the above. However, you may alternatively install it directly on UNIX. Install `OpenMP` library on UNIX as follows:

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

.. _install-openblas:

OpenBLAS (`Optional`)
---------------------

|project| can be compiled with and without OpenBLAS. If you are compiling |project| with OpenBLAS, install OpenBLAS library by

.. tab-set::

   .. tab-item:: Ubuntu/Debian
      :sync: ubuntu

      .. prompt:: bash

            sudo apt-get install libopenblas-dev

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

    NVIDIA does not support macOS. You can install NVIDIA CUDA Toolkit on Linux and Windows only.


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

    To build |project| with CUDA, you should also set ``CUDA_HOME``, ``USE_CUDA``, and optionally set ``CUDA_DYNAMIC_LOADING`` environment variabls as described in :ref:`Configure Compile-Time Environment Variables <config-env-variables>`.

Load CUDA Compiler on GPU Cluster (`Optional`)
----------------------------------------------

This section is relevant if you are using GPU on a cluster, and skip this section otherwise.

On a GPU cluster, chances are the CUDA Toolkit is already installed. If the cluster uses ``module`` interface, load CUDA as follows.

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

        These variables are relevant only if you are compiling with CUDA compiler. :ref:`Install CUDA Toolkit <install-cuda>` and specify the home directory of CUDA Toolkit by setting either of these variables. The home directory should be a path containing the executable ``/bin/nvcc`` (or ``\bin\nvcc.exe`` on Windows). For instance, if ``/usr/local/cuda/bin/nvcc`` exists, export the following:

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

        This variable is relevant only if you are compiling with CUDA compiler. By default, this variable is set to `0`. To compile |project| with CUDA, :ref:`install CUDA Toolkit <install-cuda>` and set this variable to `1` by

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

        This variable is relevant only if you are compiling with CUDA compiler. By default, this variable is set to `0`.  When |project| is complied with CUDA, the CUDA runtime libraries bundle with the final installation of |project| package, making it over 700MB. While this is generally not an issue for most users, often a small package is preferable if the installed package has to be distributed to other machines. To this end, enable the custom-made `dynamic loading` feature of |project|. In this case, the CUDA libraries will not bundle with the |project| installation, rather, |project| is instructed to load the existing CUDA libraries of the host machine at runtime. To enable dynamic loading, make sure :ref:`CUDA Toolkit <install-cuda>` is installed, then set this variable to `1` by

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

        By default, this variable is set to `0`, in which the compilation process generates source files in outside of the source directry, in ``/build`` directry. When it is set to `1`, the build files are generated in source directory. To set this variable, run

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

            Do not use this option to build the package for `production` (release) as it has a slower performance. Building the package by enabling this variable is only sitable for generting the documentation.

        .. hint::

            By enabling this variable, the build will be `in-source`, similar to setting ``CYTHON_BUILD_IN_SOURCE=1``. To clean the source directory from the generated files, see :ref:`Clean Compilation Files <clean-files>`.

    ``USE_CBLAS``

        By default, this variable is set to `0`. Set this variable to `1` if you want to use OpenBLAS instead of the built-in library of |project|. :ref:`Install OpenBLAS <install-openblas>` and set by

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

Get the source code of |project| from the Github repository by

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

    Do not load imate if your current working directory is the root directory of the source code of |project|, since python cannot load the installed package properly. Always change the current direcotry to somewhere else (for example, ``cd ..`` as shown in the above).

.. _clean-files:
   
.. rubric:: Cleaning Compilation Files

If you set ``CYTHON_BUILD_IN_SOURCE`` or ``CYTHON_BUILD_FOR_DOC`` to ``1``, the output files of Cython's compiler will be generated inside the source code directories. To clean the source code from these files (`optional`), run the following:

.. prompt:: bash

    python setup.py clean
