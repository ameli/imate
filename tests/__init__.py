"""
Some of the test files produce mstplotlib figures. During the tests (particularly on server), non of the figures should be plotted in interactive mode. Rather, the figures should be saved into ``SVG`` file. To do so, the ``/TraceInv/Utilities/PlotUtilities.py`` handles this as follows.

* If the environment variable ``DISPLAY`` does not exist or is ``''``, or
* If the environment variable ``TRACEINV_NO_DISPLAY`` is set to ``'True'``,

then, the ``matplotlib`` is imported with ``'agg'`` mode, which prevents the interactive plot, and instead, the plots are saved as ``SVG`` files.

Currently, the following test files produce figures:

    * ``test_GenerateMatrix.py``
    * ``test_InterpolateTraceOfInverse.py``.

In the beginning of the above test files, and before importing ``TraceInv``, these lines should be added:

::

    import os
    os.environ['TRACEINV_NO_DISPLAY'] = 'True'

However, the above approach yet does not prevent the interactive plotting when *all* test files are tested using ``pytest``. When the command

::

    pytest

is used, it runs all test files, including those that do not produce figure. Because those test files do not set the environment variable ``TRACEINV_NO_DISPLAY``, the ``matplotlib`` in the module ``PlotUtilities.py`` will be loaded without the ``agg`` mode, which then affects the other test files that define such environment variable, since the ``matplotlib`` is already loaded.

To prevent this issue, either the above environment variable should be loaded on *all* test files (regardless they produce figures or not), or to define it here in this ``__init__.py`` file.
"""

# For plotting matrix, we disable interactive display
import os
os.environ['TRACEINV_NO_DISPLAY'] = 'True'  # This should be before importing TraceInv packages
