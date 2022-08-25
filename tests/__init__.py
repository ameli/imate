# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


"""
Some of the test files produce mstplotlib figures. During the tests (especially
on a server), non of the figures should be plotted in interactive mode. Rather,
the figures should be saved into ``SVG`` file. To do so, the utility file
``/imate/Utilities/PlotUtilities.py`` handles this as follows:

* If the environment variable ``DISPLAY`` does not exist or is ``''``, or
* If the environment variable ``IMATE_NO_DISPLAY`` is set to ``'True'``,

then, the ``matplotlib`` is imported with ``'agg'`` mode, which prevents the
interactive plot, and instead, the plots are saved as ``SVG`` files.

Currently, the following test files produce figures:

    * ``test_correlation_matrix.py``
    * ``test_interpolate_traceinv.py``.

In the beginning of the above test files, and before importing ``imate``,
these lines should be added:

::

    import os
    os.environ['IMATE_NO_DISPLAY'] = 'True'

However, the above approach yet does not prevent the interactive plotting when
*all* test files are tested using ``pytest``. When the command

::

    pytest

is used, it runs all test files, including those that do not produce figure.
Because those test files do not set the environment variable
``IMATE_NO_DISPLAY``, the ``matplotlib`` in the module ``plot_utilities.py``
will be loaded without the ``agg`` mode, which then affects the other test
files that define such environment variable, since the ``matplotlib`` is
already loaded.

To prevent this issue, either the above environment variable should be loaded
on *all* test files (regardless they produce figures or not), or to define it
here in this ``__init__.py`` file.
"""

# For plotting matrix, we disable interactive display
import os
os.environ['IMATE_NO_DISPLAY'] = 'True'   # define before importing imate
