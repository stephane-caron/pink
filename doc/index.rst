:github_url: https://github.com/stephane-caron/pink/tree/main/doc/index.rst

.. title:: Table of Contents

####
Pink
####

**P**\ ython **in**\ verse **k**\ inematics for articulated robot models, based on `Pinocchio <https://github.com/stack-of-tasks/pinocchio>`_.

.. image:: https://user-images.githubusercontent.com/1189580/192318997-ed7574c3-8238-451d-9548-a769d46ec03b.png
   :alt: Banner for Pink

Inverse kinematics in Pink is defined by weighted :ref:`tasks <Tasks>` and :ref:`limits <Limits>`. The library adds a :ref:`configuration <Configuration>` type to Pinocchio, a configuration being a robot model and data to which forward kinematics have been applied. Given a configuration, tasks and a time step, :func:`pink.solve_ik.solve_ik` computes joint velocities that steer the model towards fulfilling all tasks at best.

.. toctree::
    :maxdepth: 1

    installation.rst
    introduction.rst
    tasks.rst
    limits.rst
    barriers.rst
    inverse-kinematics.rst
    developer-notes.rst
    references.rst

