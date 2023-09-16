.. _troubleshooting:

Troubleshooting
***************

Issue of Initializing ``libomp``
================================

When using any of the |project| functions, specifically with the ``method=slq`` option, you may encounter the following error:

::

    OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized.
    OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into
    the program. That is dangerous, since it can degrade performance or cause incorrect
    results. The best thing to do is to ensure that only a single OpenMP runtime is linked
    into the process, e.g. by avoiding static linking of the OpenMP runtime in any library.
    As an unsafe, unsupported, undocumented workaround you can set the environment variable
    KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may
    cause crashes or silently produce incorrect results. For more information, please see
    http://openmp.llvm.org/
    Abort trap: 6

.. note::

    This error is specific to *macOS* and is associated with the challenge of handling multiple copies of OpenMP runtime libraries. It is not specific to the |project| package itself.

**Workaround Attempt:**

The above error message suggests a workaround to set the environment variable ``KMP_DUPLICATE_LIB_OK=TRUE``. However, using this workaround may lead to the following error:

.. code-block::

    Segmentation fault: 11

**Solution:**

To resolve this issue, follow these steps:

1. If you have previously set the ``KMP_DUPLICATE_LIB_OK`` environment variable to ``TRUE``, unset it by running the following command:

   .. prompt:: bash
   
       unset KMP_DUPLICATE_LIB_OK

2. Next, disable or remove the Math Kernel Library by:

   .. prompt:: bash
   
       conda install nomkl
       conda remove mkl mkl-service

By following these steps, you should be able to resolve the mentioned errors when using the ``slq`` method on macOS.
