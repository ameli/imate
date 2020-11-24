*************
Running Tests
*************

The package can be tested by running the `test scripts <https://github.com/ameli/TraceInv/tree/master/tests>`_, which tests all `sub-packages <https://github.com/ameli/TraceInv/tree/master/TraceInv>`_ and the `examples <https://github.com/ameli/TraceInv/tree/master/examples>`_.

=====================
Running Tests Locally
=====================

To run a test locally, clone the source code from the repository and install the required test packages by

::

    git clone https://github.com/ameli/TraceInv.git
    cd TraceInv
    python -m pip install -e .[test]

Then, run a test with ``pytest``:

::

    pytest

To run a test coverage:

::

    pytest --cov=tests/
   
===============
Automated Tests
===============

|travis-devel| |codecov-devel|

The latest status of *automated tests* can be checked on `travis <https://travis-ci.com/github/ameli/TraceInv>`_ continuous integration tool, which tests the package in the following platforms:

====================  =======================
Platform              Python versions
====================  =======================
Linux (Ubuntu 18.04)  2.7, 3.5, 3.6, 3.7, 3.8
macOS (xcode 11)      2.7, 3.7
Windows               3.8
====================  =======================

Moreover, the latest *coverage* of tests can be checked on `codecov <https://codecov.io/gh/ameli/TraceInv>`_ dashboard.

.. |travis-devel| image:: https://img.shields.io/travis/com/ameli/TraceInv
   :target: https://travis-ci.com/github/ameli/TraceInv
.. |codecov-devel| image:: https://img.shields.io/codecov/c/github/ameli/TraceInv
   :target: https://codecov.io/gh/ameli/TraceInv
