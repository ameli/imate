% SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
% SPDX-License-Identifier: BSD-3-Clause
% SPDX-FileType: SOURCE
%
% This program is free software: you can redistribute it and/or modify it under
% the terms of the license found in the LICENSE.txt file in the root directory
% of this source tree.

% Usage:
% 
% 1. Download *.mat file from: https://sparse.tamu.edu/Janna/Queen_4147
% 2. Run this funciton with the name of the *.mat file as input argument:
%
%    >> read_matrix('Queen_4147.mat');
%
%   This script generates the following files:
%
%   * ``Queen_4147_i.mat``: Row indices        uint64
%   * ``Queen_4147_j.mat``: Column indices     uint64
%   * ``Queen_4147_v.mat``: Data               float64
%
% 3. Run the script read_matrix.py as follows:
%
%    $ read_matrix.py Queen_4147 float32    # for 32-bit data
%    $ read_matrix.py Queen_4147 float64    # for 64-bit data
%    $ read_matrix.py Queen_4147 float128   # for 128-bit data

function read_matrix(filename)
  
    load(filename);
    A = Problem.A;

    % ``A`` is a sparse matrix. Extract its non-zero row and column indices and
    % the corresponding data by:
    [i, j, v] = find(A);

    % In the above, ``i`` is the row indices, ``j`` is the column indices, and
    % ``v`` is the data. Convert ``i`` and ``j`` from double type to integr by:
    i = uint64(i);
    j = uint64(j);

    % Save them in files. Note, since the variable sizes are more than 2GB, the
    % option ``-v7.3`` is neccessary to write tem as HDF format.
    save('Queen_4147_i.mat', 'i', '-v7.3');
    save('Queen_4147_j.mat', 'j', '-v7.3');
    save('Queen_4147_v.mat', 'v', '-v7.3');

end
