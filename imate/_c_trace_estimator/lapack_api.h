/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _C_TRACE_ESTIMATOR_LAPACK_API_H_
#define _C_TRACE_ESTIMATOR_LAPACK_API_H_

extern "C"
{
    // lapack sstev
    void lapack_sstev(char* jobz, int* n, float* d, float* e, float* z,
                      int* ldz, float* work, int* info);

    // lapack dstev
    void lapack_dstev(char* jobz, int* n, double* d, double* e, double* z,
                      int* ldz, double* work, int* info);

    // lapack sdbsdc
    void lapack_sbdsdc(char* uplo, char* compq, int* n, float* d, float *e,
                       float* u, int* ldu, float* vt, int* ldvt, float* q,
                       int* iq, float* work, int* iwork, int* info);

    // lapack dbdsdc
    void lapack_dbdsdc(char* uplo, char* compq, int* n, double* d,
                       double *e, double* u, int* ldu, double* vt, int* ldvt,
                       double* q, int* iq, double* work, int* iwork,
                       int* info);
}

// lapack xstev (float overload)
template <typename DataType>
void lapack_xstev(char* jobz, int* n, DataType* d, DataType* e, DataType* z,
        int* ldz, DataType* work, int* info);

// lapack xbdsdc (float overload)
template <typename DataType>
void lapack_xbdsdc(char* uplo, char* compq, int* n, DataType* d, DataType *e,
        DataType* u, int* ldu, DataType* vt, int* ldvt, DataType* q,
        int* iq, DataType* work, int* iwork, int* info);

#endif  // _C_TRACE_ESTIMATOR_LAPACK_API_H_
