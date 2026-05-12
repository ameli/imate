#!/usr/bin/env python

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
import platform
import multiprocessing
import subprocess


# =====================
# get avail num threads
# =====================

def get_avail_num_threads():
    """
    Finds the number of CPU threads that is granted to the current user. This
    may not be all CPU threads on the machine, for instance, when the user
    requested a certain number of threads when submitting jobs to SLURM or
    Torque workload managers.

    Suppose on a machine with 8 threads, a SLURM job with --cpus-per-task=5
    is submitted.

    This function finds these quantities:

        a  = multiprocessing.cpu_count()  (here 8)    (all os)
        b  = $(nproc)                     (here 5)    (unix only)
        c  = num affinity                 (here 5)    (unit only)
        s1 = SLURM_CPUS_PER_TASK          (here 5)    (slurm only)
        s2 = SLURM_CPUS_ON_NODE           (here 5)    (slurm only)
        t1 = PBS_NUM_PPN                  (here 0)    (torque only)

    We define avail num thread as: min(a, b, max(1, s1, s2, t1)).
    """

    avail_num_threads = multiprocessing.cpu_count()

    # Num processors (unix only)
    try:
        # nproc might need to be installed on macos
        if platform.system() in ["Linux", "Darwin"]:
            nproc_output = subprocess.check_output(['nproc'], text=True)
            nproc = int(nproc_output.strip())
            if nproc < avail_num_threads:
                avail_num_threads = nproc
    except Exception:
        pass

    # Check number of available processors using affinity
    if hasattr(os, 'sched_getaffinity'):
        num_affinity = len(os.sched_getaffinity(0))
        if num_affinity < avail_num_threads:
            avail_num_threads = num_affinity

    # Query whether the number of threads are limited by SLURM, Torque, etc.
    querying_num_threads = []

    # SLURM CPUs per task
    slurm_cpus_per_task = os.getenv('SLURM_CPUS_PER_TASK')
    if slurm_cpus_per_task is not None:
        # For heterogeneous computing the number of cpu is a list
        slurm_cpus_per_task_list = \
            [int(cpu) for cpu in slurm_cpus_per_task.split(',')]
        querying_num_threads += slurm_cpus_per_task_list

    # SLURM CPUs on node
    slurm_cpus_on_node = os.getenv('SLURM_CPUS_ON_NODE')
    if slurm_cpus_on_node is not None:
        # For heterogeneous computing the number of cpu is a list
        slurm_cpus_on_node_list = \
            [int(cpu) for cpu in slurm_cpus_on_node.split(',')]
        querying_num_threads += slurm_cpus_on_node_list

    # Torque number of processes per task
    pbs_num_ppn = os.getenv('PBS_NUM_PPN')
    if pbs_num_ppn is not None:
        querying_num_threads.append(int(pbs_num_ppn))

    # Find maximum of the query
    if len(querying_num_threads) > 0:
        max_querying_num_threads = max(querying_num_threads)

        # The max of query can be a candidate if it is more than one thread
        if ((max_querying_num_threads > 0) and
                (max_querying_num_threads < avail_num_threads)):

            # Max of query should not be more than the actual number of threads
            avail_num_threads = max_querying_num_threads

    return avail_num_threads
