========
Examples
========

The scripts provided in |examplesdir|_ directory reproduces the figures and tables presented in [1]_.

---------------
Install Package
---------------

Before running examples, install ``imate`` package either by

::

	pip install imate

or if you have Anaconda (Miniconda), install by

::

	conda install -c s-ameli imate

For further installation details, see `install <https://ameli.github.io/imate/install.html>`_ page from the package documentation.

---------
Example 1
---------

To reproduce *Figure 2* of the manuscript, run the script |example1|_ by

::

    python plot_chebyshev_rational.py

The script generates the figure below.

.. image:: https://raw.githubusercontent.com/ameli/imate/main/docs/source/_static/images/interpolation/chebyshev.svg
   :align: center
   :width: 500

---------
Example 2
---------

To reproduce *Figure 3* of the manuscript, run the script |example2|_ by

::

    python imate_mwe.py

The script generates the figure below (may take a long time to complete).

.. image:: https://raw.githubusercontent.com/ameli/imate/main/docs/source/_static/images/interpolation/imate_mwe.svg
   :align: center

---------
Example 3
---------

To reproduce *Figure 4* of the manuscript, run the script |example3|_ by

::

    python plot_traceinv_full_rank.py

The script generates the three figures below.

.. image:: https://raw.githubusercontent.com/ameli/imate/main/docs/source/_static/images/interpolation/traceinv_full_rank_p0.svg
   :align: center
.. image:: https://raw.githubusercontent.com/ameli/imate/main/docs/source/_static/images/interpolation/traceinv_full_rank_p1.svg
   :align: center
.. image:: https://raw.githubusercontent.com/ameli/imate/main/docs/source/_static/images/interpolation/traceinv_full_rank_p2.svg
   :align: center

---------
Example 4
---------

To reproduce *Figure 5* of the manuscript, run the script |example4|_ by

::

    python plot_traceinv_ill_conditioned.py

The script generates the figure below.

.. image:: https://raw.githubusercontent.com/ameli/imate/main/docs/source/_static/images/interpolation/traceinv_ill_conditioned.svg
   :align: center

---------
Example 5
---------

To reproduce *Figure 6* of the manuscript, run the script |example5|_ by

::

    python plot_generalized_cross_validation.py

The script generates the figure below.

.. image:: https://raw.githubusercontent.com/ameli/imate/main/docs/source/_static/images/interpolation/generalized_cross_validation.svg
   :align: center
   :scale: 50

---------
Example 6
---------

To reproduce *Table 2* of the manuscript, run the script |example6|_ by

::

    python table_generalized_cross_validation.py

The script generates the table below.

::

    -----------------------------------------------------------------------------------------
                    Iterations   Process Time          Results            Relative Error
                    -----------  --------------  -------------------  -----------------------
    Method      q   N_tr  N_tot  T_tr    T_tot   V         log_theta  log_theta  beta   yhat
    ----------  --  ----  -----  ------  ------  -------   ---------  ---------  -----  -----
    cholesky     0   282    282   27.75   31.16  0.16376    -3.81645       0.00   0.00   0.00
    cholesky     1     3    364    0.30    4.69  0.16352    -3.56277       6.65  29.71  17.59
    cholesky     2     5    282    0.52    3.93  0.16372    -3.84457       0.74   3.69   1.95
    cholesky     3     7    284    0.69    4.12  0.16376    -3.82179       0.14   0.71   0.37
    
    hutchinson   0   334    334   60.73   64.80  0.16374    -3.94912       3.48  16.22   8.72
    hutchinson   1     3    364    0.54    4.95  0.16352    -3.56277       6.65  29.71  17.59
    hutchinson   2     5    282    0.99    4.39  0.16374    -3.84457       0.74   3.69   1.95
    hutchinson   3     7    284    1.25    4.67  0.16376    -3.77707       1.03   5.16   2.76
    
    slq          0   326    326   66.96   89.47  0.16374    -3.85426       0.99   4.93   2.61
    slq          1     3    364    0.64    5.06  0.16352    -3.56277       6.65  29.71  17.59
    slq          2     5    284    1.00    4.47  0.16374    -3.86021       1.15   5.68   3.01
    slq          3     7    282    1.35    4.83  0.16374    -3.82738       0.29   1.45   0.76

**Notes:**

* The process times shown in the above table may differ as they depend on the machine.
* The results of *hutchinson* and *SLQ* methods might differ after each run, since they are stochastic estimation methods based on Monte-Carlo sampling.
* If during the run, the error

  ::

     rational_polynomial has positive poles.

  occurred, rerun the script again, or change the location of interpolating points in the code to produce desired results.


----------
References
----------

.. [1] Ameli, S., and Shadden. S. C. (2022). Interpolating Log-Determinant and Trace of the Powers of Matrix **A** + t **B**. *Statistics and Computing* 32, 108. `DOI <https://doi.org/10.1007/s11222-022-10173-4>`__, `arXiv <https://arxiv.org/abs/2009.07385>`__.

.. |examplesdir| replace:: ``/examples`` 
.. _examplesdir: https://github.com/ameli/imate/blob/main/examples

.. |example1| replace:: ``/examples/plot_chebyshev_rational.py``
.. _example1: https://github.com/ameli/imate/blob/main/examples/plot_chebyshev_rational.py

.. |example2| replace:: ``/examples/imate_mwe.py``
.. _example2: https://github.com/ameli/imate/blob/main/examples/imate_mwe.py

.. |example3| replace:: ``/examples/plot_traceinv_full_rank.py``
.. _example3: https://github.com/ameli/imate/blob/main/examples/plot_traceinv_full_rank.py

.. |example4| replace:: ``/examples/plot_traceinv_ill_conditioned.py``
.. _example4: https://github.com/ameli/imate/blob/main/examples/plot_traceinv_ill_conditioned.py

.. |example5| replace:: ``/examples/plot_generalized_corss_validation.py``
.. _example5: https://github.com/ameli/imate/blob/main/examples/plot_generalized_cross_validation.py

.. |example6| replace:: ``/examples/table_generalized_corss_validation.py``
.. _example6: https://github.com/ameli/imate/blob/main/examples/table_generalized_cross_validation.py
