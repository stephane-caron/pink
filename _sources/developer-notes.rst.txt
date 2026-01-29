:github_url: https://github.com/stephane-caron/pink/tree/main/doc/developer-notes.rst

***************
Developer notes
***************

This section documents internal functions and other notes shared between contributors to this project.

Design guidelines
=================

* Pink is designed for clarity before performance
* Exceptions raised by the library all derive from a Pink exception base class
  to avoid abstraction leakage. See this `design decision
  <https://github.com/getparthenon/parthenon/wiki/Design-Decision:-Throw-Custom-Exceptions>`__
  for more details on the rationale behind this choice.
* Task representation strings:
    * Only report parameters that have an effect (for instance, the damping
      task does not report its :code:`lm_damping` since its error is always
      zero).
    * Parent-class attributes come after the class's own.
    * A *de facto* practice has emerged where we define `__repr__` at the
      bottom of task Python source files.

Exceptions
==========

.. automodule:: pink.exceptions
    :members:
