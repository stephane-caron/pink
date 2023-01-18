:github_url: https://github.com/tasts-robots/pink/tree/master/doc/src/installation.rst

************
Installation
************

From PyPI
=========

Installation from the Python Package Index should work via:

.. code:: bash

    pip install pin-pink

From source
===========

If installation from PyPI doesn't work, for instance on Raspberry Pi, install the following dependencies from source:

- `eigenpy <https://github.com/stack-of-tasks/eigenpy>`__ (required by Pinocchio)
- `pinocchio <https://github.com/stack-of-tasks/pinocchio>`__

Then install Pink by:

.. code:: bash

    pip install --no-deps pin-pink

Or by copying its source directory somewhere in your ``PYTHONPATH``.
