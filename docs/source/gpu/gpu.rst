.. _imate-gpu:

Using GPU Devices
*****************

|project| can run on `CUDA-capable` GPU devices with the following installed:

1. NVIDIA graphic driver,
2. CUDA libraries.

.. rubric:: CUDA Version

The version of CUDA libraries installed on the user's machine should match the version of the CUDA libraries that |project| package was compiled with. This includes matching both *major* and *minor* parts of the version numbers. However, the version's *patch* numbers do not need to be matched.

.. note::

    The |project| package that is installed with either ``pip`` or ``conda`` already has built-in support for CUDA Toolkit. The latest version of |project| is compatible with **CUDA 12.2.x**, which should match the CUDA version installed on the user's machine.

.. topic:: Methods of Setting up CUDA and |project|

    There are three ways to use |project| with a compatible version of CUDA Toolkit:

    1. :ref:`Install NVIDIA CUDA Toolkit <gpu-install-cuda>` with a CUDA version compatible with an existing |project| installation. In this way, you can keep the |project| package that is already installed with ``pip`` or ``conda``.
    2. :ref:`Compile imate from the source <gpu-compile-source>` for a specific version of CUDA to use an existing CUDA library. In this way, you can keep the current CUDA installation.
    3. :ref:`Use docker image <gpu-docker>` with pre-installed |project|, CUDA libraries, and NVIDIA graphic driver. This is the most convenient way as no compilation or installation of |project| and CUDA Toolkit is required.

The above methods are described in the following sections:

.. toctree::
    :hidden:

    self

.. toctree::
    :caption: Installing imate on GPU
    :numbered:
    
    Install NVIDIA CUDA Toolkit <gpu_install_cuda>
    Compile imate from Source with CUDA <gpu_compile_source>
    Use imate Docker Container on GPU <gpu_docker>

Once you have set up |project| on GPU devices, you can refer to the following sections to learn how to execute |project| functions on GPUs:

.. toctree::
    :caption: Using imate on GPU
    :numbered:

    Inquiry GPU and CUDA with imate <inquiry_gpu>
    Run imate Functions on GPU <run_functions_gpu>
    Deploy imate on GPU Clusters <gpu_cluster>
