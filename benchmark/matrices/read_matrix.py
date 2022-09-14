#! /usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.

"""
How to obtain the original dataset:
    https://sparse.tamu.edu/Janna/Queen_4147
    Download the matlab file

How to convert the ``Queen_4147.mat`` file:

    When the ``Queen_4147.mat`` file is loaded in matlab, a struct with the
    name ``Problem`` is created. Set

    >> load Queen_4147.mat;
    >> A = Problem.A;

    ``A`` is a sparse matrix. Extract its non-zero row and column indices and
    the corresponding data by:

    >> [i, j, v] = find(A);

    In the above, ``i`` is the row indices, ``j`` is the column indices, and
    ``v`` is the data. Convert ``i`` and ``j`` from double type to integr by:

    >> i = uint64(i);
    >> j = uint64(j);

    Save them in files. Note, since the variable sizes are more than 2GB, the
    option ``-v7.3`` is neccessary to write tem as HDF format.

    >> save('Queen_4147_i.mat', 'i', '-v7.3');
    >> save('Queen_4147_j.mat', 'j', '-v7.3');
    >> save('Queen_4147_v.mat', 'v', '-v7.3');

How to convert the mat files into scipy variable:

    Once the above three files are ready, this python script reads the three
    files:

    * ``Queen_4147_i.mat``: Row indices        uint64
    * ``Queen_4147_j.mat``: Column indices     uint64
    * ``Queen_4147_v.mat``: Data               float64

    and creates a ``scipy.sparse.csr_matrix`` object. The output is saved as a
    pickle object with filename ``Queen_4147.pickle``.
"""

# =======
# Imports
# =======

import sys
import h5py
import numpy
import os
from os.path import join
import scipy.sparse
import pickle


# ====
# main
# ====

def main(argv):
    """
    Usage:

        read_matrix.py directory dtype

    Example

        read_matrix.py Queen_4147 float32
    """

    if len(argv) < 3:
        raise ValueError('Usage: %s directory dtype' % argv[0])

    # Name of the subdirectory containing the mat file
    basename = argv[1]
    dtype = argv[2]

    directory = os.getcwd()
    filename_i = join(directory, basename, basename + '_i.mat')
    filename_j = join(directory, basename, basename + '_j.mat')
    filename_v = join(directory, basename, basename + '_v.mat')

    # Loading
    print('Loading ...')
    fi = h5py.File(filename_i, 'r')
    fj = h5py.File(filename_j, 'r')
    fv = h5py.File(filename_v, 'r')

    # Allocating
    print('Allocating ...')
    i = numpy.empty(shape=fi['i'].shape, dtype=fi['i'].dtype)
    j = numpy.empty(shape=fj['j'].shape, dtype=fj['j'].dtype)
    v = numpy.empty(shape=fv['v'].shape, dtype=fv['v'].dtype)

    # Reading
    print('Reading ...')
    fi['i'].read_direct(i)
    fj['j'].read_direct(j)
    fv['v'].read_direct(v)

    # Squeeze
    print('Squeezing ...')
    i = numpy.squeeze(i)
    j = numpy.squeeze(j)
    v = numpy.squeeze(v)

    # Convert from matlab indices to python's 0-based indices
    i = i - 1
    j = j - 1

    # Cast down
    print('Casting down ...')
    i = i.astype('uint32')
    j = j.astype('uint32')
    v = v.astype(dtype)  # Options: float32, float64, float128

    print('')
    print('Row indices [i]:')
    print(i)
    print(i.dtype)
    print(i.shape)

    print('\nColumn indices [j]:')
    print(j)
    print(j.dtype)
    print(j.shape)

    print('\nNonzero data [v]:')
    print(v)
    print(v.dtype)
    print(v.shape)
    print('')

    shape = (int(i[-1]+1), int(j[-1]+1))
    print('Shape:')
    print(shape)
    print('')

    # COO Sparse
    print('COO sparse ...')
    A_coo = scipy.sparse.coo_matrix((v, (i, j)), shape=shape, dtype=v.dtype)

    # CSR Sparse
    print('CSR sparse ...')
    A_csr = A_coo.tocsr()

    print('\nCSR matrix:')
    print('nnz: %d' % A_csr.nnz)
    print(A_csr.shape)

    # Writing
    print('Writing ...')
    output_filename = join(directory, basename, basename + '_' + str(v.dtype) +
                           '.pickle')
    with open(output_filename, 'wb') as file:
        pickle.dump(A_csr, file)
    print('Wrote to :%s.' % output_filename)


# ===========
# script main
# ===========

if __name__ == "__main__":
    sys.exit(main(sys.argv))
