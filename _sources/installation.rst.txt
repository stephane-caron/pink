:github_url: https://github.com/stephane-caron/pink/tree/main/doc/installation.rst

************
Installation
************

From Conda
==========

For best performance we recommended installing Pink from Conda:

.. code:: bash

    conda install -c conda-forge pink

From PyPI
=========

Installation from the Python Package Index should work via:

.. code:: bash

    pip install pin-pink

From source
===========

If installation from PyPI doesn't work, you can manually install `eigenpy <https://github.com/stack-of-tasks/eigenpy>`__ (required by Pinocchio) and `Pinocchio <https://github.com/stack-of-tasks/pinocchio>`__ from source. Then install Pink by:

.. code:: bash

    pip install --no-deps pin-pink

Alternatively, copy or add the ``pink/`` module directory somewhere in your ``PYTHONPATH``.
