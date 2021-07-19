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
import scipy.special

try:
    from .._utilities.plot_utilities import matplotlib, plt
    from .._utilities.plot_utilities import load_plot_settings, save_plot
    plot_modules_exist = True
except ImportError:
    plot_modules_exist = False

__all__ = ['plot_convergence']


# ===============
# remove outliers
# ===============

def _remove_outliers(
        samples,
        samples_processed_order,
        num_samples_used,
        outlier_confidence_level):
    """
    Flags the outlier samples and sets them to nan. The outlier is defined
    by those sample points that are outside of the confidence region
    defined by a confidence level.

    :param samples: 2D array of size ``(max_num_samples, num_inquiry)``. Each
        column is the set of all samples for an inquiry. Some of these samples
        might be nan, because, often, the convergence is achieved sooner than
        ``max_num_samples``, hence, with the fewer number of processed rows,
        the remaining elements in a row remain nan.
    :type: numpy.ndarray

    :param samples_processed_order: Due to parallel processing, the rows of the
        ``samples`` array are not processed sequentially. This 1D array of size
        ``max_num_samples`` keeps the record of the process order of the rows
        of ``samples``.
    :type samples_process_order: numpy.ndarray

    :param num_samples_used: 1D array of size ``num_inquiry``. The j-th
        element is the number of valid (non-nan) rows of the j-th column of
        ``samples``. Thus, it indicates how many samples exists for the j-th
        inquiry out of the max number of samples.
    :type num_samples_used: numpy.ndarray

    :param outlier_confidence_level: The confidence level of the outlier, which
        is between ``0`` and ``1``. For example, the confidence level of
        ``0.001`` means that the outliers are in the 1% area of the both ends
        of the tails of the normal distribution. When this value is zero, no
        outlier will be detected and removed. A larger level will flag more
        points as outlier.
    :type confidence_level: float

    :return: Returns two arrays:
        * ``cleaned_smaples``: an array copied from ``samples``, but the
          outliers are converted to ``numpy.nan``.
        * ``outlier_indices``: a boolean array of the same shape as ``samples``
          where the flagged outliers are set to True.
    : rtype: tuple (numpy.ndarray)
    """

    num_inquiries = samples.shape[1]
    outlier_indices = numpy.zeros(samples.shape, dtype=bool)
    cleaned_samples = numpy.copy(samples)

    # Quantile of the region that is "not" considered as outlier
    non_outlier_confidence_level = 1.0 - outlier_confidence_level
    outlier_quantile = \
        numpy.sqrt(2) * scipy.special.erfinv(non_outlier_confidence_level)

    for j in range(num_inquiries):

        sample_indices = samples_processed_order[:num_samples_used[j]]
        valid_samples = samples[sample_indices, j]

        # Mean and std of the j-th column of samples
        mean = numpy.nanmean(valid_samples)
        std = numpy.nanstd(valid_samples)

        # Find those samples outside of the confidence region
        difference = numpy.abs(samples[:, j] - mean)
        outlier_indices[:, j] = \
            (difference > outlier_quantile * std).astype(bool)

        # Remove outliers
        cleaned_samples[outlier_indices[:, j], j] = numpy.nan

    return cleaned_samples, outlier_indices


# =============================
# compute cumulative statistics
# =============================

def _compute_cumulative_statistics(
        samples,
        samples_processed_order,
        num_samples_used,
        confidence_level):
    """
    Computes mean and error as the number of samples progresses. The output
    are the means, absolute and relative errors of the samples.

    The cumulative here refers to the progressive mean or error of the samples
    from the first to the current row as the row progresses by introducing more
    rows to the samples data. For example, the i-th row and j-th column of the
    cumulative mean is the mean of rows 0:i-1 and j-th column of samples.

    :param samples: 2D array of size ``(max_num_samples, num_inquiry)``. Each
        column is the set of all samples for an inquiry. Some of these samples
        might be nan, because, often, the convergence is achieved sooner than
        ``max_num_samples``, hence, with the fewer number of processed rows,
        the remaining elements in a row remain nan.
    :type: numpy.ndarray

    :param samples_processed_order: Due to parallel processing, the rows of the
        ``samples`` array are not processed sequentially. This 1D array of size
        ``max_num_samples`` keeps the record of the process order of the rows
        of ``samples``.
    :type samples_process_order: numpy.ndarray

    :param num_samples_used: 1D array of size ``num_inquiry``. The j-th
        element is the number of valid (non-nan) rows of the j-th column of
        ``samples``. Thus, it indicates how many samples exists for the j-th
        inquiry out of the max number of samples.
    :type num_samples_used: numpy.ndarray

    :param confidence_level: The confidence level of the error, which is
        between ``0`` and ``1``. For example, the confidence level of ``0.95``
        means that the confidence region consists of 95% of the area under the
        normal distribution of the samples.
    :type confidence_level: float

    :return: A list of cumulative mean, absolute error, and relative error.
        Each of these three arrays have the same shape as ``samples``.
    :rtype: list
    """

    # Allocate arrays
    cumulative_mean = numpy.empty_like(samples)
    cumulative_abs_error = numpy.empty_like(samples)
    cumulative_rel_error = numpy.empty_like(samples)

    # Set arrays to nan
    cumulative_mean[:, :] = numpy.nan
    cumulative_abs_error[:, :] = numpy.nan
    cumulative_rel_error[:, :] = numpy.nan

    # Quantile based on the confidence level
    quantile = numpy.sqrt(2) * scipy.special.erfinv(confidence_level)

    num_inquiries = samples.shape[1]
    for j in range(num_inquiries):
        for i in range(num_samples_used[j]):

            # Find the non-nan elements of the j-th column of samples from the
            # first to the i-th processed element
            sample_indices = samples_processed_order[:i+1]
            samples_valid = samples[sample_indices, j]
            samples_valid = samples_valid[~numpy.isnan(samples_valid)]

            if samples_valid.size == 0:

                # This might happen on the first processed row
                cumulative_mean[i, j] = numpy.nan
                cumulative_abs_error[i, j] = numpy.nan
                cumulative_rel_error[i, j] = numpy.nan
            else:

                # Compute mean and errors
                cumulative_mean[i, j] = numpy.nanmean(samples_valid)
                standard_deviation = numpy.nanstd(samples_valid)
                cumulative_abs_error[i, j] = \
                    quantile * standard_deviation / numpy.sqrt(i+1)
                cumulative_rel_error[i, j] = \
                    cumulative_abs_error[i, j] / cumulative_mean[i, j]

    return cumulative_mean, cumulative_abs_error, cumulative_rel_error


# ============
# plot samples
# ============

def _plot_samples(
        ax,
        samples,
        samples_processed_order,
        num_samples_used,
        confidence_level,
        cumulative_mean,
        cumulative_abs_error):
    """
    When ``num_inquiries`` is ``1``, this function is called to plot the
    samples and their cumulative mean. When ``num_inquiries`` is more than one,
    the samples of different inquiries cannot be plotted in the same axes since
    the value of trace might be very difference for each case.

    :param ax: axes object
    :type ax: numpy.ndarray

    :param samples: 2D array of size ``(max_num_samples, num_inquiry)``. Each
        column is the set of all samples for an inquiry. Some of these samples
        might be nan, because, often, the convergence is achieved sooner than
        ``max_num_samples``, hence, with the fewer number of processed rows,
        the remaining elements in a row remain nan.
    :type: numpy.ndarray

    :param samples_processed_order: Due to parallel processing, the rows of the
        ``samples`` array are not processed sequentially. This 1D array of size
        ``max_num_samples`` keeps the record of the process order of the rows
        of ``samples``.
    :type samples_process_order: numpy.ndarray

    :param num_samples_used: 1D array of size ``num_inquiry``. The j-th
        element is the number of valid (non-nan) rows of the j-th column of
        ``samples``. Thus, it indicates how many samples exists for the j-th
        inquiry out of the max number of samples. Note that this is different
        than ``num_samples_processed``, because the j-th column might converge
        sooner than others and thus, less number of samples were used for it.
    :type num_samples_used: numpy.ndarray

    :param confidence_level: The confidence level of the error, which is
        between ``0`` and ``1``. For example, the confidence level of ``0.95``
        means that the confidence region consists of 95% of the area under the
        normal distribution of the samples.
    :type confidence_level: float

    :param cumulative_mean: 2D array of size ``(max_num_samples, num_inqiries).
    :type cumulative_mean: numpy.ndarray

    :param cumulative_abs_error: 2D array of size ``(max_num_samples,
        num_inqiries).
    :type cumulative_abs_error: numpy.ndarray
    """

    if not plot_modules_exist:
        raise ImportError('Cannot import modules for plotting. Either ' +
                          'install "matplotlib" and "seaborn" packages, ' +
                          'or set "plot=False".')

    # Load plot settings
    try:
        load_plot_settings()
    except ImportError:
        raise ImportError('Cannot import modules for plotting. Either ' +
                          'install "matplotlib" and "seaborn" packages, ' +
                          'or set "plot=False".')

    # If samples has multiple columns, only the first column (inquiry) plotted.
    inquiry = 0

    # abscissa
    x = 1 + numpy.arange(num_samples_used[inquiry])

    # Reorder rows of samples in the same order as they were processed
    rows_order = samples_processed_order[:num_samples_used[inquiry]]
    _samples = samples[rows_order, inquiry]

    # No need to reorder the rows of cumulative mean and error, since, when
    # they were computed in :func:`compute_cumulative_statistics` function,
    # they were reordered already.
    _mean = cumulative_mean[:num_samples_used[inquiry], inquiry]
    _abs_error = cumulative_abs_error[:num_samples_used[inquiry], inquiry]

    # Plot samples as points
    ax.plot(x, _samples,  'o', markersize=4, color='grey', label='estimates')

    # Plot cumulative mean
    ax.plot(x, _mean, color='black', label='mean')

    # Plot confidence region
    ax.fill_between(x, _mean - _abs_error, _mean + _abs_error, color='black',
                    alpha=0.25,
                    label=('%s' % str(100.0*confidence_level).strip('.0')) +
                    r'$\%$ confidence region')

    ax.set_xlabel('sample index')
    ax.set_ylabel('trace estimates')
    ax.set_title('Stochastic Estimates and Mean of Samples')
    ax.set_xlim([x[0], x[-1]])
    ax.legend()

    # Ensure x ticks are integers
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


# ==========
# plot error
# ==========

def _plot_error(
        ax,
        num_samples_used,
        min_num_samples,
        max_num_samples,
        error_atol,
        error_rtol,
        converged,
        cumulative_abs_error,
        cumulative_rel_error):
    """
    Plots cumulative errors. When ``num_inquiries`` is ``1``, it plots both
    absolute and relative errors (on left and right y axes) on the same axes.
    However, when ``num_inquiries`` is more than one, it only plots the
    relative error. This is because, with the multiple number of inquiries, the
    absolute error of each can be on different plot scales and they cannot be
    contained in the same scale of the plot.

    :param ax: axes object
    :type ax: numpy.ndarray

    :param num_samples_used: 1D array of size ``num_inquiry``. The j-th
        element is the number of valid (non-nan) rows of the j-th column of
        ``samples``. Thus, it indicates how many samples exists for the j-th
        inquiry out of the max number of samples. Note that this is different
        than ``num_samples_processed``, because the j-th column might converge
        sooner than others and thus, less number of samples were used for it.
    :type num_samples_used: numpy.ndarray

    :param min_num_samples: Minimum number of samples. In the iterations below
        this number, the convergence is now checked.
    :type min_num_samples: int

    :param max_num_samples: Maximum number of samples. This is also the number
        of row of ``samples`` array.
    :type max_num_samples: int

    :param error_atol: Absolute error tolerance
    :type error_atol: float

    :param error_rtol: Relative error tolerance
    :type error_rtol: float

    :param converged: 1D boolean array of size ``num_inquiries`` and indicates
        which of the inquiries are converged at their final state.
    :type converged: numpy.ndarray

    :param cumulative_abs_error: 2D array of size ``(max_num_samples,
        num_inqiries).
    :type cumulative_abs_error: numpy.ndarray

    :param cumulative_rel_error: 2D array of size ``(max_num_samples,
        num_inqiries).
    :type cumulative_rel_error: numpy.ndarray
    """

    if not plot_modules_exist:
        raise ImportError('Cannot import modules for plotting. Either ' +
                          'install "matplotlib" and "seaborn" packages, ' +
                          'or set "plot=False".')

    # Load plot settings
    try:
        load_plot_settings()
    except ImportError:
        raise ImportError('Cannot import modules for plotting. Either ' +
                          'install "matplotlib" and "seaborn" packages, ' +
                          'or set "plot=False".')

    relative_color = 'black'
    absolute_color = 'darkgrey'

    num_inquiries = cumulative_abs_error.shape[1]

    if num_inquiries == 1:
        ax2 = ax.twinx()

    # Scale the twin y axis (for absolute error) so that the final value of
    # the relative and absolute plots match
    max_rel_error = numpy.nanmax(numpy.nanmax(cumulative_rel_error)) * 100.0
    final_abs_error = cumulative_abs_error[num_samples_used[0]-1, 0]
    final_rel_error = cumulative_rel_error[num_samples_used[0]-1, 0] * 100.0

    # Match the high of the final value of the absolute error with relative
    ylim_abs_plot = final_abs_error * max_rel_error / final_rel_error

    # Extra space for the y axis limit
    ylim_scale = 1.1

    x_min = 1
    x_max = numpy.max(num_samples_used)

    # Colormaps
    if num_inquiries > 1:

        # Concatenate two colormaps, in total, 18 colors
        colors1 = matplotlib.cm.tab10.colors
        colors2 = matplotlib.cm.Dark2.colors
        colors = numpy.r_[colors1, colors2]

    else:
        colors = ['black']

    for inquiry in range(num_inquiries):

        # Abscissa
        x = 1 + numpy.arange(num_samples_used[inquiry])

        # Get relative and absolute errors for each inquiry
        _num_samples_used = num_samples_used[inquiry]
        _rel_error = cumulative_rel_error[:_num_samples_used, inquiry] * 100.0
        _abs_error = cumulative_abs_error[:_num_samples_used, inquiry]

        # With more than one inquiry, do not plot absolute error
        if num_inquiries == 1:
            ax2.plot(x, _abs_error, color=absolute_color,
                     label='absolute error')

            ax.plot(x, _rel_error, color=relative_color,
                    label='relative error')
        else:
            ax.plot(x, _rel_error, color=colors[inquiry],
                    label='inquiry: %d' % (inquiry+1))

    # Relative error tolerance limit line
    ax.plot([x_min, max_num_samples], [error_rtol, error_rtol],
            '--', color=relative_color, label='relative error tol')

    if num_inquiries == 1:

        # Absolute error tolerance limit line
        ax2.plot([x_min, max_num_samples], [error_atol, error_atol],
                 '--', color=absolute_color, label='absolute error tol')

    # Vertical dotted line showing where the min_num_samples starts
    ax.plot([min_num_samples, min_num_samples], [0., max_rel_error*ylim_scale],
            ':', color='grey', label='min num samples')

    # Plot a dot at each converged point
    dot_label_inserted = False
    for inquiry in range(num_inquiries):
        x = 1 + numpy.arange(num_samples_used[inquiry])
        _num_samples_used = num_samples_used[inquiry]
        _rel_error = cumulative_rel_error[:_num_samples_used, inquiry] * 100.0
        _abs_error = cumulative_abs_error[:_num_samples_used, inquiry]

        if converged[inquiry]:

            # Insert a label for dot only once
            if not dot_label_inserted:
                dot_label_inserted = True
                ax.plot(x[-1], _rel_error[-1], 'o', color=colors[inquiry],
                        zorder=20, markersize=4, label='convergence reached')
            else:
                ax.plot(x[-1], _rel_error[-1], 'o', color=colors[inquiry],
                        zorder=20, markersize=4)

    # Before min_num_samples and after max_num_samples, shade plot background
    ax.axvspan(x_min, min_num_samples, color='grey', alpha=0.2, lw=0,
               label='convergence skipped')
    ax.axvspan(x_max, max_num_samples, color='grey', alpha=0.07, lw=0)

    ax.set_xlabel('sample index')
    ax.set_ylabel('relative error')
    ax.set_title('Convergence of Errors')
    ax.set_xlim([x_min, max_num_samples])
    ax.set_ylim([0, max_rel_error*ylim_scale])

    # Display yticks as percent
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())

    # Ensure x ticks are integers
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    if num_inquiries == 1:
        ax2.set_ylabel('absolute error')
        ax2.set_ylim([0, ylim_abs_plot*ylim_scale])

    # Legend
    lines, labels = ax.get_legend_handles_labels()
    if num_inquiries == 1:
        # With one inquiry, legend contains both relative and absolute plots
        # where each correspond to the left and right (twin) axis
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines = lines + lines2
        all_labels = labels + labels2
        ax.legend(all_lines, all_labels)

    elif num_inquiries < 11:
        # With less than 11 inquiries, use single column legend
        ax.legend(lines, labels, bbox_to_anchor=(1.01, 1.022),
                  loc='upper left')
    else:
        # With equal or more than 11 inquiries, use two column legend
        ax.legend(lines, labels, bbox_to_anchor=(1.01, 1.022),
                  loc='upper left', ncol=2)

    # Bring the plots of the original axis above the plots of the twin axis
    ax.set_zorder(1)
    ax.patch.set_visible(False)


# ================
# plot convergence
# ================

def plot_convergence(info):
    """
    Plots samples, cumulative mean, absolute and relative error.

    :param info: A dictionary of all output info.
    :type: dict
    """

    if not plot_modules_exist:
        raise ImportError('Cannot import modules for plotting. Either ' +
                          'install "matplotlib" and "seaborn" packages, ' +
                          'or set "plot=False".')

    # Load plot settings
    try:
        load_plot_settings()
    except ImportError:
        raise ImportError('Cannot import modules for plotting. Either ' +
                          'install "matplotlib" and "seaborn" packages, ' +
                          'or set "plot=False".')

    # Extract variables from info dictionary
    num_inquiries = info['matrix']['num_inquiries']

    error_atol = info['error']['error_atol']
    error_rtol = info['error']['error_rtol'] * 100.0
    confidence_level = info['error']['confidence_level']

    num_samples_used = info['convergence']['num_samples_used']
    min_num_samples = info['convergence']['min_num_samples']
    max_num_samples = info['convergence']['max_num_samples']
    converged = info['convergence']['converged']
    samples = info['convergence']['samples']
    samples_processed_order = info['convergence']['samples_processed_order']

    # Convert scalars to arrays for easier indexing. When num_inquiries is 1,
    # all variables below are scalars. These are converted to arrays of size 1.
    if numpy.isscalar(num_samples_used):
        num_samples_used = numpy.array([num_samples_used])
    if numpy.isscalar(converged):
        converged = numpy.array([converged])
    if samples.ndim == 1:
        samples = numpy.array([samples]).T
    if numpy.isscalar(converged):
        converged = numpy.array([converged])

    # Remove outliers from samples
    outlier_confidence_level = 0.001
    cleaned_samples, outlier_indices = _remove_outliers(
            samples,
            samples_processed_order,
            num_samples_used,
            outlier_confidence_level)

    # Compute cumulative mean and errors (as sample indices progress)
    cumulative_mean, cumulative_abs_error, cumulative_rel_error = \
        _compute_cumulative_statistics(
                cleaned_samples,
                samples_processed_order,
                num_samples_used,
                confidence_level)

    # Different plots depending on number of inquiries
    if num_inquiries == 1:

        # Plot both samples (on left axis) and relative error (on right axis)
        fig, ax = plt.subplots(ncols=2, figsize=(11, 5))
        _plot_samples(ax[0], cleaned_samples, samples_processed_order,
                      num_samples_used, confidence_level, cumulative_mean,
                      cumulative_abs_error)
        _plot_error(ax[1], num_samples_used, min_num_samples, max_num_samples,
                    error_atol, error_rtol, converged, cumulative_abs_error,
                    cumulative_rel_error)

    else:

        # Plot only relative error (one axis) but for all inquiries
        fig, ax = plt.subplots(ncols=1, figsize=(8.3, 5))
        _plot_error(ax, num_samples_used, min_num_samples, max_num_samples,
                    error_atol, error_rtol, converged, cumulative_abs_error,
                    cumulative_rel_error)

    plt.tight_layout()

    # Check if the graphical backend exists
    if matplotlib.get_backend() != 'agg':
        plt.show()
    else:
        # write the plot as SVG file in the current working directory
        save_plot(plt, 'Convergence', transparent_background=True)
