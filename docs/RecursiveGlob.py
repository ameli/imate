# =======
# Imports
# =======

import os
import fnmatch

__all__ = ['RecursiveGlob']

# =======================
# Split All Parts Of Path
# =======================

def SplitAllPartsOfPath(Path):
    """
    Splits all parts of a path. For example, the path

        '../build/lib.linux-x86_64-3.8/Module/Submodule/lib.so'

    will be split to the list

        ['..','build','lib.linux-x86_64','Module','Submodule','lib.so']

    :param Path: A file or directory path
    :type Path: string

    :return: The list of strings of split path.
    :rtype: list(string)
    """

    AllParts = []

    # Recursion
    while True:

        # Split last part
        Parts = os.path.split(Path)

        if Parts[0] == Path:
            AllParts.insert(0,Parts[0])
            break

        elif Parts[1] == Path:
            AllParts.insert(0,Parts[1])
            break

        else:
            Path = Parts[0]
            AllParts.insert(0,Parts[1])

    return AllParts

# =========================
# Remove Duplicates In List
# =========================

def RemoveDuplicatesInList(List):
    """
    Removes duplicate elements in a list.

    :param List: A list with possibly duplicate elements.
    :type List: list

    :return: A list which elements are not duplicate.
    :rtype: list
    """

    ShrinkedList = []
    for Element in List:
        if Element not in ShrinkedList:
            ShrinkedList.append(Element)

    return ShrinkedList

# ==============
# Recursive Glob
# ==============

def RecursiveGlob(Directory,Patterns):
    """
    Recursively searches all subdirectories of a given directory and looks for
    a list of patterns. If in a subdirectory, one of the patterns is found, the name of the first
    immediate subdirectory (after Directory) is returned in a list.

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

    This code outputs ['SubDirectory-1','SubDirectory-2']. Note that the `SubSubDirectory-2` is not in the output,
    since it is part of the `SubDirectory-2'. That is, this code only outputs the first children subdirectory
    if within that subdirectory a match of pattern is found.

    .. note::
        
        In python 3, this function ``glob(dir,recursive=True)`` can be simply used. However, the
        recursive glob is not supproted in python 2 version of ``glob``, hence this function is written.

    :param Directory: The path of a directory.
    :type Directory: string

    :param Patterns: A list of string as regex pattern, such as ['*.so','*.dylib','*.dll']
    :type Patterns: list(string)

    :return: List of first-depth subdirectories that within them a match of pattern is found.
    :rtype: list(string)
    """

    # Find how many directory levels are in the input Directory path
    DirectoryDepth = len(SplitAllPartsOfPath(Directory))

    SubDirectories=[]

    for Root, DirName, Filenames in os.walk(Directory):
        for Pattern in Patterns:
            for Filename in fnmatch.filter(Filenames,Pattern):
                SubDirectories.append(SplitAllPartsOfPath(Root)[DirectoryDepth])

    return RemoveDuplicatesInList(SubDirectories)
