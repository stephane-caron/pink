:github_url: https://github.com/stephane-caron/pink/tree/main/doc/introduction.rst

************
Introduction
************

Inverse kinematics (IK) is the problem of computing *motions* (in Pink: velocities) that achieve a given set of *tasks*, such as putting a foot on a surface, moving the center of mass to a target location, etc.

This documentation assumes you are already familiar with task-based inverse kinematics. You can check out for instance this `post on inverse kinematics <https://scaron.info/robot-locomotion/inverse-kinematics.html>`__ for a general introduction.

Notations
=========

In Pink, we adopt the subscript right-to-left convention for transforms, and superscript notation to indicate the frame of a motion or force vector:

.. list-table::
    :class: cheatsheet
    :widths: 70 30

    * - Quantity
      - Notation
    * - Affine transform from frame :math:`A` to frame :math:`B`
      - :math:`T_{BA}`
    * - Body angular velocity of frame :math:`A` in frame :math:`B`
      - :math:`{}^A \omega_{BA}`
    * - Position of frame :math:`B` in frame :math:`A`
      - :math:`{}^A p_B`
    * - Rotation matrix from frame :math:`A` to frame :math:`B`
      - :math:`R_{BA}`
    * - Spatial angular velocity of frame :math:`A` in frame :math:`B`
      - :math:`{}^B \omega_{BA}`
    * - World frame (inertial)
      - :math:`W`

With these notations frame transforms can be read left to right, for example:

.. raw:: latex html

    \begin{align}
    T_{CA} & = T_{CB} T_{BA} &
    {}^{B} \omega & = R_{BA} {}^{A} \omega &
    {}^B p_C & = R_{BA} {}^A p_C + {}^B p_A
    \end{align}

See also this `spatial algebra cheat sheet
<https://scaron.info/robot-locomotion/spatial-vector-algebra-cheat-sheet.html>`_.

.. _Configuration:

Configuration
=============

.. automodule:: pink.configuration
    :members:

Task formalism
==============

The *task function approach* followed in Pink, and many works before it, was originally formalized by Samson, Espiau and Le Borgne in [Samson1991]_.

.. automodule:: pink.tasks.task
    :members:
