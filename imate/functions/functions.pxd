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

cdef extern from "identity.h":

    # Identity
    cdef cppclass Identity(Function):
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil

cdef extern from "indicator.h":

    # Indicator
    cdef cppclass Indicator(Function):
        Indicator(double a_, double b_) except +
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil
        double a
        double b

cdef extern from "inverse.h":

    # Inverse
    cdef cppclass Inverse(Function):
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil

cdef extern from "homographic.h":

    # Homographic
    cdef cppclass Homographic(Function):
        Homographic(double a_, double b_, double c_, double d_) except +
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil
        double a
        double b
        double c
        double d

cdef extern from "logarithm.h":

    # Logarithm
    cdef cppclass Logarithm(Function):
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil

cdef extern from "exponential.h":

    # Exponential
    cdef cppclass Exponential(Function):
        Exponential(double coeff_) except +
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil
        double coeff

cdef extern from "exponential.h":

    # Power
    cdef cppclass Power(Function):
        Power() except +
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil
        double exponent

cdef extern from "gaussian.h":

    # Gaussian
    cdef cppclass Gaussian(Function):
        Gaussian(double mu_, double sigma_) except +
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil
        double mu
        double sigma

cdef extern from "smoothstep.h":

    # Smooth Step
    cdef cppclass SmoothStep(Function):
        SmoothStep(double alpha_) except +
        float function(const float lambda_) nogil
        double function(const double lambda_) nogil
        long double function(const long double lambda_) nogil
        double alpha
