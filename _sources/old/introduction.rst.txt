************
Introduction
************

============
Applications
============

The above function is featured in a wide range of applications in statistics and machine learning. Particular applications are in model selection and optimizing hyperparameters with gradient-based maximum likelihood methods. In such applications, computing the above function is often a computational challenge for large matrices. Often, this function is evaluated for a wide range of the parameter :math:`t` while :math:`\mathbf{A}` and :math:`\mathbf{B}` remain fixed. As such, an interpolation scheme enables fast computation of the function.

A common example of such an application can be found in regularization techniques applied to inverse problems and supervised learning. For instance, in ridge regression by generalized cross-validation (see [Golub-1997]_), the optimal regularization parameter :math:`t` is sought by minimizing a function that involves :eq:`map` (see :ref:`Example 3 <Example_Three>`). Another common usage of :eq:`map`, for instance, is the mixed covariance functions of the form :math:`\mathbf{A} + t \mathbf{I}` that appear frequently in Gaussian processes with additive noise [Ameli-2020]_ (see :ref:`Example 1 <Example_One>`). In most of these applications, the log-determinant of the covariance matrix is ubiquitous, particularly in likelihood functions or related variants. Namely, if one aims to maximize the likelihood by its derivative with respect to the parameter, the expression

.. math::

    \frac{\partial}{\partial t} \log \det (\mathbf{A} + t \mathbf{I}) = \mathrm{trace} \left( (\mathbf{A} + t \mathbf{I})^{-1} \right),

frequently appears. Other examples of :eq:`map` are in the optimal design of experiment, probabilistic principal component analysis (see Sec. 12.2 of [Bishop-2006]_), relevance vector machines [Tipping-2001]_ and [Bishop-2006]_, kernel smoothing (see Sec. 2.6 of [Rasmussen-2006]_, and Bayesian linear models (see Sec. 3.3 of [Bishop-2006]_.

==========
References
==========

.. [Ameli-2020] Ameli, S., and Shadden. S. C. (2020). Interpolating the Trace of the Inverse of Matrix :math:`\mathbf{A} + t \mathbf{B}`. `arXiv:2009.07385 <https://arxiv.org/abs/2009.07385>`__ [math.NA]

.. [Golub-1997] Golub, G. H. & von Matt, U. (1997). Generalized cross-validation for large-scale problems. Journal of Computational and Graphical Statistics, 6(1), 1-34. `doi: 10.2307/1390722 <https://www.jstor.org/stable/1390722>`_

.. [Tipping-2001] Michael E. Tipping. (2001). Sparse bayesian learning and the relevance vector machine. J. Mach. Learn. Res. 1, 211-244. `doi: 10.1162/15324430152748236 <https://dl.acm.org/doi/10.1162/15324430152748236>`_

.. [Bishop-2006] Bishop, C. M. (2006). Pattern Recognition and Machine Learning (Information Science and Statistics). Berlin, Heidelberg: Springer-Verlag, `ISBN: 978-0-387-31073-2 <https://www.springer.com/gp/book/9780387310732>`_

.. [Rasmussen-2006] Rasmussen, C. E. & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. Adaptive Computation and Machine Learning. Cambridge, MA, USA: MIT Press. `ISBN 0-262-18253-X <http://www.gaussianprocess.org/gpml/>`_.
