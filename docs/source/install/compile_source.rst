.. _compile-source:

Compile from Source
===================

.. contents::

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

You can compile |project| with any of the following compilers:

* `GCC <https://gcc.gnu.org/>`__ (Linux, macOS, Windows via `MinGW <https://www.mingw-w64.org/>`__ or `Cygwin <https://www.cygwin.com/>`__)
* `LLVM/Clang <https://clang.llvm.org/>`__ (Linux, macOS, Windows via `MinGW <https://www.mingw-w64.org/>`__, or LLVM's own Windows support) and `LLVM/Clang by Apple <https://opensource.apple.com/projects/llvm-clang/>`__ 
* `Intel OneAPI <https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html#gs.5c6ir2>`__ (Linux, Windows)
* `Microsoft Visual Studio (MSVC) Compiler for C++ <https://code.visualstudio.com/docs/cpp/config-msvc#:~:text=You%20can%20install%20the%20C,the%20C%2B%2B%20workload%20is%20checked.>`_ (Windows)
* `Arm Compiler for Linux <https://developer.arm.com/Tools%20and%20Software/Arm%20Compiler%20for%20Linux>`__ (Linux on AARCH64 architecture)

Below are short description of setting up a few major compilers:

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

Then, export ``CC`` and ``CXX`` variables by

.. prompt:: bash

  export CC=/usr/local/bin/gcc
  export CXX=/usr/local/bin/g++

.. rubric:: Install Clang/LLVN Compiler
  
.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            sudo apt install clang libomp-dev

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum install yum-utils
            sudo yum-config-manager --enable extras
            sudo yum makecache
            sudo yum install clang libomp-devel

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf install yum-utils
            sudo dnf config-manager --enable extras
            sudo dnf makecache
            sudo dnf install clang libomp-devel

    .. tab-item:: macOS
        :sync: osx

        .. prompt:: bash

            sudo brew install llvm libomp-dev

Then, export ``CC`` and ``CXX`` variables by

.. prompt:: bash

  export CC=/usr/local/bin/clang
  export CXX=/usr/local/bin/clang++

.. rubric:: Install Intel oneAPI Compiler

To install `Intel Compiler` see `Intel oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html>`__. Once installed, set the compiler's required environment variables by

.. tab-set::

    .. tab-item:: UNIX
        :sync: unix

        .. prompt:: bash

            source /opt/intel/oneapi/setvars.sh

    .. tab-item:: Windows (Powershell)
        :sync: win

        .. prompt:: powershell

            C:\Program Files (x86)\Intel\oneAPI\setvars.bat

In UNIX, export ``CC`` and ``CXX`` variables by

.. prompt:: bash

    export CC=`which icpx`
    export CXX=`which icpx`

.. _install_openmp:
   
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

    In *macOS*, for ``libomp`` versions ``15`` and above, Homebrew installs OpenMP as *keg-only*. To utilize the OpenMP installation, you should establish the following symbolic links:

    .. prompt:: bash

        libomp_dir=$(brew --prefix libomp)
        ln -sf ${libomp_dir}/include/omp-tools.h  /usr/local/include/omp-tools.h
        ln -sf ${libomp_dir}/include/omp.h        /usr/local/include/omp.h
        ln -sf ${libomp_dir}/include/ompt.h       /usr/local/include/ompt.h
        ln -sf ${libomp_dir}/lib/libomp.a         /usr/local/lib/libomp.a
        ln -sf ${libomp_dir}/lib/libomp.dylib     /usr/local/lib/libomp.dylib

.. _install-openblas:

Install OpenBLAS (`Optional`)
-----------------------------

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

          sudo brew install openblas -y

Alternatively, you can install OpenBLAS using ``conda``:

.. prompt:: bash

    conda install -c anaconda openblas

To build |project| with OpenBLAS, you should also set ``CGLAFS`` and ``LFFLAGS``, for instance by

.. tab-set::

    .. tab-item:: UNIX
        :sync: unix

        .. prompt:: bash

            CONDA_PREFIX=$(conda info --base)
            export CFLAGS="-I$CONDA_PREFIX/include $CFLAGS"
            export LDFLAGS="-L$CONDA_PREFIX/lib $LDFLAGS"

    .. tab-item:: Windows (Powershell)
        :sync: win

        .. prompt:: bash

            $env:CONDA_PREFIX = (conda info --base).Trim()
            $env:CFLAGS = "-I$env:CONDA_PREFIX\include $env:CFLAGS"
            $env:LDFLAGS = "-L$env:CONDA_PREFIX\lib $env:LDFLAGS"

You should also set ``USE_CBLAS`` environment variable as described in :ref:`Configure Compile-Time Environment Variables <config-env-variables>`.

.. _install-mkl:

Install Intel's Math Kernel Library (`Optional`)
------------------------------------------------

|project| can be compiled with and without `Intel's Math Kerel Library (MKL) <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`__. You can install MKL from `Intel oneAPI <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html>`__, or with ``conda``:

.. prompt:: bash

    conda install mkl mkl-include -y

To build |project| with MKL, you should also set ``CGLAFS`` and ``LFFLAGS``, for instance by

.. tab-set::

    .. tab-item:: UNIX
        :sync: unix

        .. prompt:: bash

            CONDA_PREFIX=$(conda info --base)
            export CFLAGS="-I$CONDA_PREFIX/include $CFLAGS"
            export LDFLAGS="-L$CONDA_PREFIX/lib $LDFLAGS"

    .. tab-item:: Windows (Powershell)
        :sync: win

        .. prompt:: bash

            $env:CONDA_PREFIX = (conda info --base).Trim()
            $env:CFLAGS = "-I$env:CONDA_PREFIX\include $env:CFLAGS"
            $env:LDFLAGS = "-L$env:CONDA_PREFIX\lib $env:LDFLAGS"

You should also set ``USE_MKL`` environment variable as described in :ref:`Configure Compile-Time Environment Variables <config-env-variables>`.

.. _install-cuda:

Install CUDA Compiler (`Optional`)
----------------------------------

To use |project| on GPU devices, it should be compiled with CUDA compiler. Skip this part if you are not using GPU.

.. note::

    The minimum version of CUDA to compile |project| is `CUDA 10.0`.

.. attention::

    NVIDIA does not support macOS. You can install the NVIDIA CUDA Toolkit on Linux and Windows only.

To download and install the CUDA Toolkit on both Linux and Windows, refer to the `NVIDIA Developer website <https://developer.nvidia.com/cuda-downloads>`__. It's important to note that NVIDIA's installation instructions on their website include the entire CUDA Toolkit, which is typically quite large (over 6 GB in size).

However, for compiling |project|, you don't need to install the entire CUDA Toolkit. Instead, only the CUDA compiler and a few specific development libraries, such as cuBLAS and cuSparse, are required. Below are simplified installation instructions for Linux, allowing you to perform a minimal CUDA installation with only the necessary libraries. Note that in the following, you may change ``CUDA_VERSION`` to the CUDA version that you wish to install.

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            # Set to the desired cuda version
            CUDA_VERSION="12-3"

            # Machine architecture
            ARCH=$(uname -m | grep -q -e 'x86_64' && echo 'x86_64' || echo 'sbsa')

            # OS Version
            UBUNTU_VERSION=$(awk -F= '/^VERSION_ID/{gsub(/"/, "", $2); print $2}' /etc/os-release)
            OS_VERSION=$(dpkg --compare-versions "$UBUNTU_VERSION" "ge" "22.04" && echo "2204" || echo "2004")

            # Add CUDA Repository 
            sudo apt update
            sudo apt install wget -y
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${OS_VERSION}/${ARCH}/cuda-keyring_1.1-1_all.deb -P /tmp
            sudo dpkg -i /tmp/cuda-keyring_1.1-1_all.deb
            rm /tmp/cuda-keyring_1.1-1_all.deb

            # Install required CUDA libraries
            sudo apt-get update
            sudo apt install -y \
                cuda-nvcc-${CUDA_VERSION} \
                libcublas-${CUDA_VERSION} \
                libcublas-dev-${CUDA_VERSION} \
                libcusparse-${CUDA_VERSION} \
                libcusparse-dev-${CUDA_VERSION}

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            # Set to the desired cuda version
            CUDA_VERSION="12-3"

            # Machine architecture
            ARCH=$(uname -m | grep -q -e 'x86_64' && echo 'x86_64' || echo 'sbsa')

            # OS Version
            OS_VERSION=$(awk -F= '/^VERSION_ID/{gsub(/"/, "", $2); print $2}' /etc/os-release)

            # Add CUDA Repository 
            sudo yum install -y yum-utils
            sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel${OS_VERSION}/${ARCH}/cuda-rhel${OS_VERSION}.repo

            # Install required CUDA libraries
            sudo yum install --setopt=obsoletes=0 -y \
                cuda-nvcc-${CUDA_VERSION} \
                cuda-cudart-devel-${CUDA_VERSION} \
                libcublas-${CUDA_VERSION} \
                libcublas-devel-${CUDA_VERSION} \
                libcusparse-${CUDA_VERSION} \
                libcusparse-devel-${CUDA_VERSION}

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            # Set to the desired cuda version
            CUDA_VERSION="12-3"

            # Machine architecture
            ARCH=$(uname -m | grep -q -e 'x86_64' && echo 'x86_64' || echo 'sbsa')

            # OS Version
            OS_VERSION=$(awk -F= '/^VERSION_ID/{gsub(/"/, "", $2); print $2}' /etc/os-release)

            # Add CUDA Repository 
            sudo dnf install -y dnf-utils
            sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel${OS_VERSION}/${ARCH}/cuda-rhel${OS_VERSION}.repo

            # Install required CUDA libraries
            sudo dnf install --setopt=obsoletes=0 -y \
                cuda-nvcc-${CUDA_VERSION} \
                cuda-cudart-devel-${CUDA_VERSION} \
                libcublas-${CUDA_VERSION} \
                libcublas-devel-${CUDA_VERSION} \
                libcusparse-${CUDA_VERSION} \
                libcusparse-devel-${CUDA_VERSION}

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

                    $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7"

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

                    $env:USE_CUDA = "1"

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

                    $env:CUDA_DYNAMIC_LOADING = "1"

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

                    $env:CYTHON_BUILD_IN_SOURCE = "1"

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

                    $env:CYTHON_BUILD_FOR_DOC = "1"

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

                    $env:USE_CBLAS = "1"

        .. note::

            Both ``USE_CBLAS`` and ``USE_MKL`` cannot be set to `1` as only one of these libraries should be used.

    ``USE_MKL``

        By default, this variable is set to `0`. Set this variable to `1` if you want to use Interl's Math Kernel Library (MKL) instead of the built-in library of |project|. :ref:`Install MKL <install-mkl>`, set ``CFLAGS`` and ``LDFLAGS`` corresponding to MKL's installation path, and set

        .. tab-set::

            .. tab-item:: UNIX
                :sync: unix

                .. prompt:: bash

                    export USE_MKL=1

            .. tab-item:: Windows (Powershell)
                :sync: win

                .. prompt:: powershell

                    $env:USE_MKL = "1"

        .. note::

            Both ``USE_MKL`` and ``USE_CBLAS`` cannot be set to `1` as only one of these libraries should be used.

    ``USE_OPENMP``
        
        To enable shared-memory parallelization uisng OpenMP, set this variable to `1` and make sure OpenMP is installed (see :ref:`Install OpenMP <install_openmp>`). Setting this variable to `0` disables this feature. By default, this variable is set to `1`.

        .. tab-set::

            .. tab-item:: UNIX
                :sync: unix

                .. prompt:: bash

                    export USE_OPENMP=1

            .. tab-item:: Windows (Powershell)
                :sync: win

                .. prompt:: powershell

                    $env:USE_OPENMP = "1"

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

                    $env:DEBUG_MODE = "1"

        .. attention::

            With the debugging mode enabled, the size of the package will be larger and its performance may be slower, which is not suitable for `production`.

    ``USE_LONG_INT``

        By default, index variables are compiled using 32-bit signed integers type. To use signed 64-bit (long int) type instead, set this option to `1` by

        .. tab-set::

            .. tab-item:: UNIX
                :sync: unix

                .. prompt:: bash

                    export USE_LONG_INT=1

            .. tab-item:: Windows (Powershell)
                :sync: win

                .. prompt:: powershell

                    $env:USE_LONG_INT = "1"

    ``USE_UNSIGNED_LONG_INT``

        By default, index variables are compiled using 32-bit signed integers type. To use unsigned 64-bit (unsigned long int) type instead, set this option to `1` by

        .. tab-set::

            .. tab-item:: UNIX
                :sync: unix

                .. prompt:: bash

                    export USE_UNSIGNED_LONG_INT=1

            .. tab-item:: Windows (Powershell)
                :sync: win

                .. prompt:: powershell

                    $env:USE_UNSIGNED_LONG_INT = "1"

Compile and Install
-------------------

|repo-size|

Get the source code of |project| from the GitHub repository by

.. prompt:: bash

    git clone https://github.com/ameli/imate.git
    cd imate

To compile and install, run

.. prompt:: bash

    python -m pip install .

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

            sudo -E python -m pip install .

    .. tab-item:: Windows (Powershell)
        :sync: win

        .. code-block:: PowerShell
            :emphasize-lines: 5

            $env:CUDA_HOME = "/usr/local/cuda"
            $env:USE_CUDA = "1"
            $env:CUDA_DYNAMIC_LOADING = "1"

            sudo -E python -m pip install .

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

.. |repo-size| image:: https://img.shields.io/github/repo-size/ameli/imate
   :target: https://github.com/ameli/imate
