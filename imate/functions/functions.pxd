# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Externs
# =======

cdef extern from "functions.h":

    # Function
    cdef cppclass Function:
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil

    # Indentity
    cdef cppclass Identity(Function):
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil

    # Inverse
    cdef cppclass Inverse(Function):
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil

    # Logarithm
    cdef cppclass Logarithm(Function):
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil

    # Exponential
    cdef cppclass Exponential(Function):
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil

    # Power
    cdef cppclass Power(Function):
        Power() except +
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil
        double exponent

    # Gaussian
    cdef cppclass Gaussian(Function):
        Gaussian() except +
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil
        double mu
        double sigma

    # Smooth Step
    cdef cppclass SmoothStep(Function):
        SmoothStep() except +
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil
        double alpha
