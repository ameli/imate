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

import os
import subprocess
import sys

# resource is not available in windows
if os.name == 'posix':
    import resource

__all__ = ['Memory']


# ======
# memory
# ======

class Memory(object):
    """
    Measures resident memory size or its change for the Python process.

    Attributes
    ----------

    ini_mem_used : default=0
        The initial resident memory when :meth:`imate.Memory.start` is called.

    mem : int default=0
        The difference between the current resident memory when
        :meth:`imate.Memory.read` is called and the initial resident memory.

    Methods
    -------
    start
    read
    get_resident_memory

    See Also
    --------

    imate.info
    imate.Timer

    Notes
    -----

    **Resident Memory:**

    The resident set size (or RSS) is the occupied memory of the current
    python process which resides on the RAM. If the memory of the current
    process overflows onto the disk's swap space, only the memory residing on
    RAM is measured by RSS.

    **How to Use:**

    * To measure the resident memory in the current Python process, call
      :meth:`imate.Memory.get_resident_memory` function. For this, the class
      :meth:`imate.Memory` does not needed to be instantiated.

    * To measure the *acquired* memory between two points of the code (that is,
      finding the *difference* of the resident memory), first instantiate the
      :class:`imate.Memory` class. Then call the two functions
      :meth:`imate.Memory.start` and :meth:`imate.Memory.read` of the
      instantiated object on the two points where the memory difference should
      be measured.

    Examples
    --------

    The following example tracks the resident memory *acquired* during the
    computation of the log-determinant of a matrix. In particular, the
    :meth:`imate.Memory.read` in this example reads the *difference* between
    the resident memory of the two lines highlighted below.

    .. code-block:: python
        :emphasize-lines: 10, 16

        >>> # Create an object of Memory class
        >>> from imate import Memory
        >>> mem = Memory()

        >>> # Create a matrix
        >>> from imate import toeplitz, logdet
        >>> A = toeplitz(2, 1, size=1000, gram=True)

        >>> # Start tracking memory change from here
        >>> mem.start()

        >>> # Compute the log-determinant of the matrix
        >>> ld = logdet(A)

        >>> # Read acquired memory is acquired from start to this point
        >>> mem.read()
        (679936, 'b')

        >>> # Or, read acquired memory in human-readable format
        >>> mem.read(human_readable=True)
        (664.0, 'Kb')

    The following example shows the current resident memory of the process
    as is at the current reading on the hardware.

    .. code-block:: python

        >>> # Load Memory module
        >>> from imate import Memory

        >>> # Get resident memory in bytes
        >>> Memory.get_resident_memory()
        (92954624, 'b')

        >>> # Get resident memory in human-readable format
        >>> Memory.get_resident_memory(human_readable=True)
        (88.6484375, 'Mb')
    """

    # ====
    # init
    # ====

    def __init__(self):
        """
        Initialization.
        """

        # Initial memory in Bytes
        self.init_mem = 0

        # Memory increase in bytes
        self.mem_diff = 0

    # =====
    # start
    # =====

    def start(self):
        """
        Sets the start points to track the memory change.

        .. note::

            This method should be called before calling
            :meth:`imate.Memory.read`.

        See Also
        --------

        imate.Memory.read

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 10

            >>> # Create an object of Memory class
            >>> from imate import Memory
            >>> mem = Memory()

            >>> # Create a matrix
            >>> from imate import toeplitz, logdet
            >>> A = toeplitz(2, 1, size=1000, gram=True)

            >>> # Start tracking memory change from here
            >>> mem.start()

            >>> # Compute the log-determinant of the matrix
            >>> ld = logdet(A)

            >>> # Read acquired memory is acquired from start to this point
            >>> mem.read()
            (679936, 'b')

            >>> # Or, read acquired memory in human-readable format
            >>> mem.read(human_readable=True)
            (664.0, 'Kb')
        """

        # Get memory usage in bytes as of the current moment.
        self.init_mem = Memory._get_resident_memory_in_bytes()

    # ====
    # read
    # ====

    def read(self, human_readable=False):
        """
        Returns the memory used in the current process.

        .. note::

            This method should be called after :py:meth:`imate.Memory.start` is
            called.

        Parameters
        ----------

        human_readable : bool, default=False
            If `False`, the output is in Bytes. If `True`, the output is
            converted to a human readable unit. The unit can be checked by
            ``Memory.mem_unit`` attribute.

        Returns
        -------

        mem_tuple : tuple (int, str)
            A tuple consists of the amount of acquired memory together with a
            string indicating the unit in which the memory is reported. If
            ``human_readable`` is `False` the unit is ``'b'`` indicating Bytes
            unit. If ``human_readable`` is `True`, other units may be used as
            follows:

            * ``"b"``: indicates Bytes
            * ``"KB"``: indicates Kilo-Bytes
            * ``"MB"``: indicates Mega-Bytes
            * ``"GB"``: indicates Giga-Bytes
            * ``"TB"``: indicates Tera-Bytes
            * ``"PB"``: indicates Peta-Bytes
            * ``"EB"``: indicates Exa-Bytes
            * ``"ZB"``: indicates Zetta-Bytes
            * ``"YB"``: indicates Yotta-Bytes

        See Also
        --------

        imate.Memory.start
        imate.Memory.get_resident_memory

        Notes
        -----

        This method reads the *difference* between the resident memory from
        when :meth:`imate.Memory.start` is called to the point where this
        method is called. Hence, this method measures the *acquired* memory
        in between two points.

        In contrast, the function :meth:`imate.Memory.get_resident_memory`
        returns the current memory that resides in the hardware.

        Examples
        --------

        .. code-block:: python
            :emphasize-lines: 16, 20

            >>> # Create an object of Memory class
            >>> from imate import Memory
            >>> mem = Memory()

            >>> # Create a matrix
            >>> from imate import toeplitz, logdet
            >>> A = toeplitz(2, 1, size=1000, gram=True)

            >>> # Start tracking memory change from here
            >>> mem.start()

            >>> # Compute the log-determinant of the matrix
            >>> ld = logdet(A)

            >>> # Read acquired memory is acquired from start to this point
            >>> mem.read()
            (679936, 'b')

            >>> # Or, read acquired memory in human-readable format
            >>> mem.read(human_readable=True)
            (664.0, 'Kb')
        """

        final_mem = Memory._get_resident_memory_in_bytes()

        # Memory increase in bytes
        self.mem_diff = final_mem - self.init_mem

        # Convert from bytes to the closest unit
        if human_readable:
            mem_diff, unit = Memory._human_readable_memory(self.mem_diff)
        else:
            mem_diff = self.mem_diff
            unit = 'b'

        return mem_diff, unit

    # ===================
    # get resident memory
    # ===================

    @staticmethod
    def get_resident_memory(human_readable=False):
        """
        Returns the resident memory of the current process.

        This method is a *static method* and does not require to instantiate
        :class:`imate.Memory` class.

        Parameters
        ----------

        human_readable : bool, default=False
            If `False`, the output is in Bytes. If `True`, the output is
            converted to a human readable unit.

        Returns
        -------

        mem_tuple : tuple (int, str)
            A tuple consists of the resident memory together with a string
            indicating the unit in which the memory is reported. If
            ``human_readable`` is `False` the unit is ``'b'`` indicating Bytes
            unit. If ``human_readable`` is `True`, other units may be used as
            follows:

            * ``"b"``: indicates Bytes
            * ``"KB"``: indicates Kilo-Bytes
            * ``"MB"``: indicates Mega-Bytes
            * ``"GB"``: indicates Giga-Bytes
            * ``"TB"``: indicates Tera-Bytes
            * ``"PB"``: indicates Peta-Bytes
            * ``"EB"``: indicates Exa-Bytes
            * ``"ZB"``: indicates Zetta-Bytes
            * ``"YB"``: indicates Yotta-Bytes

        See Also
        --------

        imate.Memory.read

        Notes
        -----
        This function returns the resident memory that currently resides in the
        hardware.

        Note that, in contrast, :meth:`imate.Memory.read` reads the
        *difference* between the resident memory from when
        :meth:`imate.Memory.start` is called to the point where this method is
        called, hence measures the *acquired* memory in between two points.

        Examples
        --------

        .. code-block:: python

            >>> # Load Memory module
            >>> from imate import Memory

            >>> # Get resident memory in bytes
            >>> Memory.get_resident_memory()
            (92954624, 'b')

            >>> # Get resident memory in human-readable format
            >>> Memory.get_resident_memory(human_readable=True)
            (88.6484375, 'Mb')
        """

        mem = Memory._get_resident_memory_in_bytes()

        # Convert from bytes to the closets unit
        if human_readable:
            mem, unit = Memory._human_readable_memory(mem)
        else:
            mem = mem
            unit = 'b'

        return mem, unit

    # ============================
    # get resident memory in bytes
    # ============================

    @staticmethod
    def _get_resident_memory_in_bytes():
        """
        Returns the resident memory in the current process.

        Returns
        -------

        mem : int
            The current resident memory in the current Python process. The
            memory is in Bytes unit.
        """

        # Convert Kb to bytes
        k = 2**10

        if os.name == 'posix':
            # In Linux and MaxOS
            mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            # In Linux, the output of the command is in Kb. Convert to Bytes.
            if sys.platform == 'linux':
                mem *= k

        else:
            # In windows
            pid = os.getpid()
            command = ['tasklist', '/fi', '"pid eq %d"' % pid]

            try:
                pid = os.getpid()
                command = ['tasklist', '/fi', 'pid eq %d' % pid]
                process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                error_code = process.poll()
                if error_code != 0:
                    mem = 'n/a'
                    return mem

                # Parse output
                last_line = stdout.strip().decode().split("\n")[-1]

                # Check last line of output has any number in it
                is_digit = [char.isdigit() for char in last_line]
                if not any(is_digit):
                    mem = 'n/a'
                    return mem

                # Get memory as string and its unit
                mem_string = last_line.split(' ')[-2].replace(',', '')
                mem = int(mem_string)
                mem_unit = last_line.split(' ')[-1]

                # Convert bytes based on the unit
                if mem_unit == 'K':
                    exponent = 1
                if mem_unit == 'M':
                    exponent = 2
                if mem_unit == 'G':
                    exponent = 3
                if mem_unit == 'T':
                    exponent = 4

                # Memory in bytes
                mem = mem * (k**exponent)

            except FileNotFoundError:
                mem = 'n/a'

        return mem

    # =====================
    # human readable memory
    # =====================

    @staticmethod
    def _human_readable_memory(mem_bytes):
        """
        Converts memory in Bytes to human readable unit.
        """

        k = 2**10
        counter = 0
        mem_hr = mem_bytes

        while mem_hr > k:
            mem_hr /= k
            counter += 1

        if counter == 0:
            unit = 'b'       # Byte
        elif counter == 1:
            unit = 'Kb'      # Kilo byte
        elif counter == 2:
            unit = 'Mb'      # Mega byte
        elif counter == 3:
            unit = 'Gb'      # Giga byte
        elif counter == 4:
            unit = 'Tb'      # Tera byte
        elif counter == 5:
            unit = 'Pb'      # Peta byte
        elif counter == 6:
            unit = 'Eb'      # Exa byte
        elif counter == 7:
            unit = 'Zb'      # Zetta byte
        elif counter == 8:
            unit = 'Yb'      # Yotta byte

        return mem_hr, unit
