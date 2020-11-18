*********
Utilities
*********

The modules in this directory provide helper class and functions to the examples in the parent directory. Namely

* ``DataUtilties.py`` generates correlation matrix ``K``, noisy data ``z``, and design matrix ``X``.
* ``PlotUtilities.py`` import all plot packages, customize the plot themes to use ``seaborn`` package, and set the plot fonts to LaTeX (if installed).
* ``ProcessingTimeUtilities.py`` defines functions to measure elapsed time, such as ``ProcessTime()`` to measure elspaded time, ``TimeCounterClass`` to store elapsed time, and ``RestrictComputationToSingleProcessor()``.
