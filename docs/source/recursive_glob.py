# SPDX-FileCopyrightText: Copyright 2022, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

import os
import fnmatch

__all__ = ['recursive_glob']


# =======================
# split all parts of path
# =======================

def _split_all_parts_of_path(path):
    """
    Splits all parts of a path. For example, the path

        '../build/lib.linux-x86_64-3.8/Module/Submodule/lib.so'

    will be split to the list

        ['..','build','lib.linux-x86_64','Module','Submodule','lib.so']

    :param path: A file or directory path
    :type path: string

    :return: The list of strings of split path.
    :rtype: list(string)
    """

    all_parts = []

    # Recursion
    while True:

        # Split last part
        parts = os.path.split(path)

        if parts[0] == path:
            all_parts.insert(0, parts[0])
            break

        elif parts[1] == path:
            all_parts.insert(0, parts[1])
            break

        else:
            path = parts[0]
            all_parts.insert(0, parts[1])

    return all_parts


# =========================
# remove duplicates in list
# =========================

def _remove_duplicates_in_list(list):
    """
    Removes duplicate elements in a list.

    :param list: A list with possibly duplicate elements.
    :type list: list

    :return: A list which elements are not duplicate.
    :rtype: list
    """

    shrinked_list = []
    for element in list:
        if element not in shrinked_list:
            shrinked_list.append(element)

    return shrinked_list


# ==============
# recursive glob
# ==============

def recursive_glob(directory, patterns):
    """
    Recursively searches all subdirectories of a given directory and looks for
    a list of patterns. If in a subdirectory, one of the patterns is found, the
    name of the first immediate subdirectory (after Directory) is returned in a
    list.

    For example, is the pattern is '*.so',

    Directory
    |
    |-- SubDirectory-1
    |   |
    |   |--File-1.so
    |
    |-- SubDirectory-2
    |   |
    |   |--SubSubDirectory-2
    |      |
    |      |--File-2.so
    |
    |-- SubDirectory-3
    |   |
    |   |--File-3.a

    This code outputs ['SubDirectory-1','SubDirectory-2']. Note that the
    `SubSubDirectory-2` is not in the output, since it is part of the
    `SubDirectory-2'. That is, this code only outputs the first children
    subdirectory if within that subdirectory a match of pattern is found.

    .. note::

        In python 3, this function ``glob(dir,recursive=True)`` can be simply
        used. However, the recursive glob is not supproted in python 2 version
        of ``glob``, hence this function is written.

    :param directory: The path of a directory.
    :type directory: string

    :param patterns: A list of string as regex pattern, such as
        ['*.so','*.dylib','*.dll']
    :type patterns: list(string)

    :return: List of first-depth subdirectories that within them a match of
        pattern is found.
    :rtype: list(string)
    """

    # Find how many directory levels are in the input Directory path
    directory_depth = len(_split_all_parts_of_path(directory))

    subdirectories = []

    for root, dirname, filenames in os.walk(directory):
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                subdirectories.append(
                        _split_all_parts_of_path(root)[directory_depth])

    return _remove_duplicates_in_list(subdirectories)
