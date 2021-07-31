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

import numpy
from scipy.special import erfinv


# =================
# check convergence
# =================

def check_convergence(
        samples,
        min_num_samples,
        processed_samples_indices,
        num_processed_samples,
        confidence_level,
        error_atol,
        error_rtol):
    """
    Checks if the standard deviation of the set of the cumulative averages of
    trace estimators converged below the given tolerance.

    The convergence criterion for each trace inquiry is if:

        standard_deviation < max(rtol * average[i], atol)

    where `error_rtol` and `error_atol` are relative and absolute tolerances,
    respectively. If this criterion is satisfied for *all* trace inquiries,
    this function returns `1`, otherwise `0`.
    """

    # If number of processed samples are not enough, set to not converged yet.
    # This is essential since in the first few iterations, the standard
    # deviation of the cumulative averages are still too small.
    if num_processed_samples < min_num_samples:

        # Skip computing error. Fill outputs with trivial initial values
        converged = 0
        num_samples_used = num_processed_samples
        return converged, num_samples_used

    # Quantile of normal distribution (usually known as the "z" coefficient)
    standard_z_score = numpy.sqrt(2) * erfinv(confidence_level)

    # mean using all processed rows of j-th column
    mean = numpy.mean(
        samples[processed_samples_indices[:num_processed_samples]])

    # std using all processed rows of j-th column
    if num_processed_samples > 1:
        std = numpy.std(
            samples[processed_samples_indices[:num_processed_samples]])
    else:
        std = numpy.inf

    # Compute error based of std and confidence level
    error = standard_z_score * std / numpy.sqrt(num_processed_samples)

    # Check error with atol and rtol to find if converged
    if error < numpy.max([error_atol, error_rtol*mean]):
        converged = 1
    else:
        converged = 0

    # Update how many samples used so far to average
    num_samples_used = num_processed_samples

    return converged, num_samples_used


# =================
# average estimates
# =================

def average_estimates(
        confidence_level,
        outlier_significance_level,
        max_num_samples,
        num_samples_used,
        processed_samples_indices,
        samples):
    """
    Averages the estimates of trace. Removes outliers and reevaluates the error
    to take into account for the removal of the outliers.

    The elimination of outliers does not affect the elements of samples array,
    rather it only affects the reevaluation of trac and error arrays.
    """

    # Flag which samples are outliers
    outlier_indices = numpy.zeros((max_num_samples, ), dtype=int)

    # Quantile of normal distribution (usually known as the "z" coefficient)
    error_z_score = numpy.sqrt(2) * erfinv(confidence_level)

    # Confidence level of outlier is the complement of significance level
    outlier_confidence_level = 1.0 - outlier_significance_level

    # Quantile of normal distribution area where is not considered as outlier
    outlier_z_score = numpy.sqrt(2.0) * erfinv(outlier_confidence_level)

    # Compute mean of samples
    mean = numpy.mean(
        samples[processed_samples_indices[:num_samples_used]])

    # Compute std of samples
    if num_samples_used > 1:
        std = numpy.std(
            samples[processed_samples_indices[:num_samples_used]])
    else:
        std = numpy.inf

    # Outlier half interval
    outlier_half_interval = outlier_z_score * std

    # Difference of each element from
    num_outliers = 0
    for i in range(num_samples_used):

        mean_discrepancy = samples[processed_samples_indices[i]] - mean
        if numpy.abs(mean_discrepancy) > outlier_half_interval:

            # Outlier detected
            outlier_indices[i] = 1
            num_outliers += 1

    # Reevaluate mean but leave out outliers
    summand = 0.0
    for i in range(num_samples_used):
        if outlier_indices[i] == 0:
            summand += samples[processed_samples_indices[i]]
    mean = summand / (num_samples_used - num_outliers)

    # Reevaluate std but leave out outliers
    if num_samples_used > 1 + num_outliers:
        summand = 0.0
        for i in range(num_samples_used):
            if outlier_indices[i] == 0:
                mean_discrepancy = samples[processed_samples_indices[i]] - mean
                summand += mean_discrepancy * mean_discrepancy

        std = numpy.sqrt(summand/(num_samples_used - num_outliers - 1.0))
    else:
        std = numpy.inf

    # trace and its error
    trace = mean
    error = error_z_score * std / numpy.sqrt(num_samples_used - num_outliers)

    return trace, error, num_outliers
