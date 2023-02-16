:github_url: https://github.com/tasts-robots/pink/tree/master/doc/src/index.rst

.. title:: Table of Contents

####
Pink
####

.. image:: images/pink-round-corners-140x140.png
   :height: 140px
   :alt: alternate text
   :align: right

**P**\ ython **in**\ verse **k**\ inematics for articulated robot models, based on `Pinocchio <https://github.com/stack-of-tasks/pinocchio>`_.

Inverse kinematics in Pink is defined by weighted :ref:`tasks <Tasks>` and :ref:`joint limits <Joint limits>`. The library adds a :ref:`configuration <Configuration>` type to Pinocchio, a configuration being a robot model and data to which forward kinematics have been applied. Given a configuration, tasks and a time step, :func:`pink.solve_ik.solve_ik` computes joint velocities that steer the model towards fulfilling all tasks at best.

.. toctree::
    :maxdepth: 1

    installation.rst
    introduction.rst
    tasks.rst
    joint-limits.rst
    inverse-kinematics.rst
    developer-notes.rst
    references.rst

You can also download this documentation as a `PDF document <pink.pdf>`_.
