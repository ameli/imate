.. _AllExamples:

********
Examples
********

Three examples are provided in |examplesdir|_, which aim to reproduce the figures presented in [Ameli-2020]_. Namely, in that reference,

* **Example 1:** Script |example1|_ reproduces Figure 2.
* **Example 2:** Script |example2|_ reproduces Figure 3.
* **Example 3:** Script |example3|_ reproduces Figure 4 and generates the results of Table 2.

Before running examples:
   To run the examples, you may not need to install the ``imate`` package. Rather, download the source code and install requirements:

   ::
    
       # Download
       git clone https://github.com/ameli/imate.git

       # Install prerequisite packages
       cd imate
       python -m pip install --upgrade -r requirements.txt
    
   Then, run either of the examples as described below.

.. _Example_One:

=========
Example 1
=========

The script |example1|_ plots the interpolation of the function

.. math::

    \tau(t) = \mathrm{trace} \left( (\mathbf{A} + t \mathbf{I})^{-1} \right)

using **Root Monomial basis Function** method. Here, :math:`\mathbf{I}` is the identity matrix, :math:`\mathbf{A}` is a dense full-rank correlation matrix of the size :math:`50^2 \times 50^2`, and :math:`t \in [10^{-4},10^3]`.

Run this example by

::

    python examples/Plot_imate_FullRank.py

The script generates the figure below (see Figure 2 of [Ameli-2020]_).

.. image:: images/Example1.svg
   :align: center

The plot on the left shows the interpolation of :math:`\tau(t)`. Each colored curve is obtained using different number of interpolant points :math:`p`. The plot on the right represents the relative error  of interpolation compared to the accurate computation when no interpolation is applied. Clearly, employing more interpolant points (such as the red curve with :math:`p = 9` interpolant points) yield smaller interpolation error.

.. _Example_Two:

=========
Example 2
=========

The script |example2|_ plots the interpolation of the function :math:`\tau(t)` as defined in :ref:`Example 1`, however, here, the matrix :math:`\mathbf{A}` is defined by 

.. math::

    \mathbf{A} = \mathbf{X}^{\intercal} \mathbf{X} + s \mathbf{I}

where :math:`\mathbf{X}` is an ill-conditioned matrix of the size :math:`1000 \times 500`, and the fixed shift parameter :math:`s=10^{-3}` is applied to improve the condition number of the matrix to become invertible.


The interpolation is performed using **Rational Polynomial Function** method for :math:`t \in [-10^{-3},10^{3}]`.


Run this example by

::

    python examples/Plot_imate_IllConditioned.py

The script generates the figure below (see also  Figure 3 of [Ameli-2020]_).

.. image:: images/Example2.svg
   :align: center

.. _Example_Three:

=========
Example 3
=========

The script |example3|_ plots the `Generalized Cross-validation <https://www.jstor.org/stable/1390722?seq=1>`_ function

.. math::

    V(\theta) = \frac{\frac{1}{n} \| \mathbf{I} - \mathbf{X} (\mathbf{X}^{\intercal} \mathbf{X} + n \theta \mathbf{I})^{-1} \mathbf{X}^{\intercal} \boldsymbol{z} \|_2^2}{\left( \frac{1}{n} \mathrm{trace}\left( (\mathbf{I} - \mathbf{X}(\mathbf{X}^{\intercal} \mathbf{X} + n \theta \mathbf{I})^{-1})\mathbf{X}^{\intercal} \right) \right)^2}

where :math:`\mathbf{X}` is the same matrix as :ref:`Example 2` and the term involving the trace of inverse in the denominator is interpolated as presented in :ref:`Example 2`.

Run this example by

::

    python examples/Plot_GeneralizedCrossValidation.py

The script generates the figure below and prints the processing times of the computations. See more details in Figure 3 and results of Table 2 of [Ameli-2020]_.

.. image:: images/GeneralizedCrossValidation.svg
   :width: 550
   :align: center


.. |examplesdir| replace:: ``/examples`` 
.. _examplesdir: https://github.com/ameli/imate/blob/main/examples
.. |example1| replace:: ``/examples/Plot_imate_FullRank.py``
.. _example1: https://github.com/ameli/imate/blob/main/examples/Plot_imate_FullRank.py
.. |example2| replace:: ``/examples/Plot_imate_IllConditioned.py``
.. _example2: https://github.com/ameli/imate/blob/main/examples/Plot_imate_IllConditioned.py
.. |example3| replace:: ``/examples/Plot_GeneralizedCorssValidation.py``
.. _example3: https://github.com/ameli/imate/blob/main/examples/Plot_GeneralizedCrossValidation.py
