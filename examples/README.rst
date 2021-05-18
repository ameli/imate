********
Examples
********

Three examples are provided in |examplesdir|_, which aim to reproduce the figures presented in [Ameli-2020]_. Namely, in that reference,

Before running examples:
   To run the examples, you may not need to install the ``imate`` package. Rather, download the source code and install requirements:

   ::
    
       # Download
       git clone https://github.com/ameli/imate.git

       # Install prerequisite packages
       cd imate
       python -m pip install --upgrade -r requirements.txt
    
   Then, run either of the examples as described below.


=========
Example 1
=========

Run the script |example1|_ by

::

    python examples/Plot_imate_FullRank.py

The script generates the figure below (see Figure 2 of [Ameli-2020]_).

.. image:: https://raw.githubusercontent.com/ameli/imate/main/docs/images/Example1.svg
   :align: center

=========
Example 2
=========

Run the script |example2|_ by

::

    python examples/Plot_imate_IllConditioned.py

The script generates the figure below (see also  Figure 3 of [Ameli-2020]_).

.. image:: https://raw.githubusercontent.com/ameli/imate/main/docs/images/Example2.svg
   :align: center

=========
Example 3
=========

Run the script |example3|_ by

::

    python examples/Plot_GeneralizedCrossValidation.py

The script generates the figure below and prints the processing times of the computations. See more details in Figure 3 and results of Table 2 of [Ameli-2020]_.

.. image:: https://raw.githubusercontent.com/ameli/imate/main/docs/images/GeneralizedCrossValidation.svg
   :width: 550
   :align: center

**********
References
**********

.. [Ameli-2020] Ameli, S., and Shadden. S. C. (2020). Interpolating the Trace of the Inverse of Matrix **A** + t **B**. `arXiv:2009.07385 <https://arxiv.org/abs/2009.07385>`__ [math.NA]

.. |examplesdir| replace:: ``/examples`` 
.. _examplesdir: https://github.com/ameli/imate/blob/main/examples
.. |example1| replace:: ``/examples/Plot_imate_FullRank.py``
.. _example1: https://github.com/ameli/imate/blob/main/examples/Plot_imate_FullRank.py
.. |example2| replace:: ``/examples/Plot_imate_IllConditioned.py``
.. _example2: https://github.com/ameli/imate/blob/main/examples/Plot_imate_IllConditioned.py
.. |example3| replace:: ``/examples/Plot_GeneralizedCorssValidation.py``
.. _example3: https://github.com/ameli/imate/blob/main/examples/Plot_GeneralizedCrossValidation.py
