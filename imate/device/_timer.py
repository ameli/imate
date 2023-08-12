# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import time

# Check python version
import sys
if sys.version_info[0] == 2:
    python2 = True
else:
    python2 = False


# =====
# Timer
# =====

class Timer(object):
    """
    A timer to measure elapsed wall time and CPU process time of Python
    process.

    Parameters
    ----------

    hold : bool, default=True
        When ``hold`` is `True`, measuring time between successive `tic-toc`
        calls are cumulative.

    Attributes
    ----------

    init_wall_time : float, default=0
        The initial wall time in seconds. This is set by
        :meth:`imate.Timer.tic`.

    init_proc_time : float, default=0
        The initial CPU process time in seconds. This is set by
        :meth:`imate.Timer.tic`.

    tic_initiated : bool, default=False
        Indicates whether the function :meth:`imate.Timer.tic` has been called
        to initiate tracking time.

    hold : bool
        Initialized by the argument ``hold``, and indicates whether to
        accumulate the recording of time between successive tic-toc calls.

    wall_time : float, default=0
        The wall time between a `tic` and a `toc` in seconds. This variable is
        updated by calling :meth:`imate.Timer.toc`.

    proc_time : float, default=0
        The CPU process time between a `tic` and a `toc` in seconds. This
        variable is updated by calling :meth:`imate.Timer.toc`.

    count : int, default=0
        Counts how many pairs of `tic-toc` is called. That is, the counter
        counts a completed pair of `tic-toc` calls. If there is one tic but
        multiple toc calls later, this is counted as only once.

    Methods
    -------
    tic
    toc
    reset

    See Also
    --------

    imate.Memory
    imate.device.restrict_to_single_processor

    Notes
    -----

    **Difference Between Wall and Process Times:**

    The *wall* time (``Timer.wall_time``) measures the wall's clock time in
    between the execution of two tasks, including. This includes when the
    processor is idle or performs other tasks other the Python process.

    On the other hand, the *process* time (``Timer.proc_time``) is the combined
    CPU clock on all cores of the CPU processor of the process. It excludes the
    time when the processor is idle. Namely, it only measures how much the
    processor was busy. Also, it only measures the time it takes to run the
    *current* process, and not other tasks that the process might perform
    concurrently.

    As a rule of thumb, the process time is larger than the wall time by the
    order of the number of CPU cores (see
    :func:`imate.device.get_num_cpu_threads`). However, the process time can
    in some cases deviate from this rule significantly (or even be less than
    the wall time if the processor was largely idle).

    Generally, the process time is the preferred measure to benchmark a
    computational task.

    Examples
    --------

    **Using Tic-Toc:**

    A simple usage of measuring wall and process time to compute the
    log-determinant of a sample matrix:

    .. code-block:: python
        :emphasize-lines: 8, 16

        >>> # Load Timer class
        >>> from imate import Timer

        >>> # Instantiate a timer object
        >>> timer = Timer()

        >>> # Start tracking time
        >>> timer.tic()

        >>> # do something time-consuming
        >>> from imate import toeplitz, logdet
        >>> A = toeplitz(2, 1, size=1000000, gram=True)
        >>> ld, info = logdet(A, method='slq', return_info=True)

        >>> # Register a time-stamp right here in the code
        >>> timer.toc()

        >>> # Read wall time
        >>> timer.wall_time
        2.727652072906494

        >>> # Read process time
        >>> timer.proc_time
        17.752098541000002

    The `toc` calls can continue (multiple `toc` calls). In each of the `toc`
    calls, the measured time is updated from the time s `toc` is called
    with respect to the last `tic` call.

    **Alternative Way of Measuring Time Using Function Returns:**

    In the above example we also passed ``return_info=True`` argument to the
    :func:`imate.logdet` function, which returns the dictionary ``info``.
    The key ``info['time']`` also keeps the track of computation time of this
    function, which can be compared with the wall and process times  measured
    by :class:`imate.Timer` as follows:

    .. code-block:: python

        >>> info['time']
        {
            'tot_wall_time': 2.617882580962032,
            'alg_wall_time': 2.5974619388580322,
            'cpu_proc_time': 17.642682527999998
        }

    **Resetting Timer:**

    The above timer can be reset as follows:

    .. code-block:: python

        >>> # Reset timer
        >>> timer.reset()
        17.752098541000002

        # By resetting, all attributes of the object set back to zero
        >>> timer.wall_time
        0.0

        >>> time.proc_time
        0.0

    **Accumulative Time Measure Using Hold:**

    Often it is useful to measure times accumulatively, in between multiple
    pairs of `tic-toc` calls. To do so, set ``hold`` to `True` as follows. The
    measured time in the example below is the addition of the first two and the
    second two calls to the pair of `tic` and `toc` on the highlighted lines,
    which includes the time to process matrices `A` and `C`, but excludes the
    matrix `B`.

    .. code-block:: python
        :emphasize-lines: 8, 16, 24, 32

        >>> # Load Timer class
        >>> from imate import Timer

        >>> # Instantiate a timer object
        >>> timer = Timer(hold=True)

        >>> # The first call to tic
        >>> timer.tic()

        >>> # The first task with matrix A.
        >>> from imate import toeplitz, logdet
        >>> A = toeplitz(2, 1, size=1000000, gram=True)
        >>> logdet(A, method='slq')

        >>> # The first call to toc stops tracking time since last tic.
        >>> timer.toc()

        >>> # A second task with matrix B. Elapsed time for this task is not
        >>> # recorded.
        >>> B = toeplitz(3, 2, size=1000000, gram=True)
        >>> logdet(B, method='slq')

        >>> # Resume tracking time.
        >>> timer.tic()

        >>> # The third task with matrix C. Time for this job is added to the
        >>> # time tracking since timer is resumed.
        >>> C = toeplitz(4, 3, size=1000000, gram=True)
        >>> logdet(C, method='slq')

        >>> # The second call to toc stops tracking time since last tic.
        >>> timer.toc()

        >>> # Read wall time
        >>> timer.wall_time
        5.431022644042969

        >>> # Read process time
        >>> timer.proc_time
        36.046730618
    """

    # ====
    # init
    # ====

    def __init__(self, hold=True):
        """
        Initialization.
        """

        # Internal variable used to store initial timestamps
        self.init_wall_time = 0.0
        self.init_proc_time = 0.0
        self.tic_initiated = False
        self.hold = hold

        # Public attributes
        self.wall_time = 0.0
        self.proc_time = 0.0
        self.count = 0

    # ===
    # tic
    # ===

    def tic(self):
        """
        Initializes tracking time.

        .. note::

            This function should be called before calling
            :meth:`imate.Timer.toc`.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 8

            >>> # Load Timer class
            >>> from imate import Timer

            >>> # Instantiate a timer object
            >>> timer = Timer()

            >>> # Start tracking time
            >>> timer.tic()

            >>> # do something time-consuming
            >>> from imate import toeplitz, logdet
            >>> A = toeplitz(2, 1, size=1000000, gram=True)
            >>> ld, info = logdet(A, method='slq', return_info=True)

            >>> # Register a time-stamp right here in the code
            >>> timer.toc()

            >>> # Read wall time
            >>> timer.wall_time
            2.727652072906494

            >>> # Read process time
            >>> timer.proc_time
            17.752098541000002
        """

        self.init_wall_time = time.time()

        if python2:
            self.init_proc_time = time.time()
        else:
            self.init_proc_time = time.process_time()

        # This variable is used to count a complete tic-toc call.
        self.tic_initiated = True

    # ===
    # toc
    # ===

    def toc(self):
        """
        Measures time from the last call to :meth:`imate.Timer.tic`.

        .. note::

            This function should be called after calling
            :meth:`imate.Timer.tic`.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 16

            >>> # Load Timer class
            >>> from imate import Timer

            >>> # Instantiate a timer object
            >>> timer = Timer()

            >>> # Start tracking time
            >>> timer.tic()

            >>> # do something time-consuming
            >>> from imate import toeplitz, logdet
            >>> A = toeplitz(2, 1, size=1000000, gram=True)
            >>> ld, info = logdet(A, method='slq', return_info=True)

            >>> # Register a time-stamp right here in the code
            >>> timer.toc()

            >>> # Read wall time
            >>> timer.wall_time
            2.727652072906494

            >>> # Read process time
            >>> timer.proc_time
            17.752098541000002
        """

        wall_time_ = time.time() - self.init_wall_time

        if python2:
            proc_time_ = time.time() - self.init_proc_time
        else:
            proc_time_ = time.process_time() - self.init_proc_time

        if self.hold:
            # Cumulative time between successive tic-toc
            self.wall_time += wall_time_
            self.proc_time += proc_time_
        else:
            # Only measures the elapsed time for the current tic-toc
            self.wall_time = wall_time_
            self.proc_time = proc_time_

        # Prevents counting multiple toc calls which were initiated with one
        # tic call.
        if self.tic_initiated:
            self.tic_initiated = False
            self.count += 1

    # =====
    # reset
    # =====

    def reset(self):
        """
        Resets time counters.

        This function is used when an instance of this class should be
        reused again from fresh settings.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 19

            >>> # Load Timer class
            >>> from imate import Timer

            >>> # Instantiate a timer object
            >>> timer = Timer()

            >>> # Start tracking time
            >>> timer.tic()

            >>> # do something time-consuming
            >>> from imate import toeplitz, logdet
            >>> A = toeplitz(2, 1, size=1000000, gram=True)
            >>> ld, info = logdet(A, method='slq', return_info=True)

            >>> # Register a time-stamp right here in the code
            >>> timer.toc()

            >>> # Read wall time
            >>> timer.wall_time
            2.727652072906494

            >>> # Read process time
            >>> timer.proc_time
            17.752098541000002

            >>> # Reset timer
            >>> timer.reset()

            # By resetting, all attributes of the object set back to zero
            >>> timer.wall_time
            0.0

            >>> time.proc_time
            0.0
        """

        self.wall_time = 0.0
        self.proc_time = 0.0
        self.tic_initiated = False
        self.count = 0
