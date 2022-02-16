#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Stéphane Caron.
#
# This file is part of Pink.
#
# Pink is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Pink is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Pink. If not, see <http://www.gnu.org/licenses/>.

"""
All kinematic tasks derive from the following base class. The formalism used in
this implementation is written down in `this note on task-based inverse
kinematics <https://scaron.info/robotics/inverse-kinematics.html>`_. As of
February 2022 it hasn't been updated with the proper dimensional analysis, but
the core concepts and notations are there.
"""

from typing import Tuple

import numpy as np
import pinocchio as pin


class Task:

    """
    Base class for kinematic tasks.

    Attributes:
        gain: Task gain :math:`\\alpha \\in [0, 1]` for additional low-pass
            filtering. Defaults to 1.0 (no filtering) for dead-beat control.
    """

    gain: float = 1.0

    def compute_task_dynamics(
        self, robot: pin.RobotWrapper
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the matrix :math:`J(q)` and vector :math:`\\alpha e(q)` such
        that the task dynamics are:

        .. math::

            J(q) \\Delta q = \\alpha e(q)

        The Jacobian matrix is :math:`J(q) \\in \\mathbb{R}^{k \\times n_v}`,
        with :math:`n_v` the dimension of the robot's tangent space and
        :math:`k` the dimension of the task. The error vector :math:`e(q) \\in
        \\mathbb{R}^k` is multiplied by the task gain :math:`\\alpha \\in [0,
        1]`. The gain is usually 1 for dead-beat control (*i.e.* converge as
        fast as possible), but it can also be lower for some extra low-pass
        filtering.

        Both :math:`J(q)` and :math:`(e)` depend on the configuration :math:`q`
        of the robot. The configuration displacement :math:`\\Delta q` is the
        output of inverse kinematics.

        Args:
            robot: Robot model and its current configuration :math:`q`.

        Returns:
            Tuple :math:`(J, e)` of Jacobian matrix and error vector, both
            expressed in the body frame.
        """
        raise NotImplementedError

    def compute_qp_objective(
        self, robot: pin.RobotWrapper
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the matrix-vector pair :math:`(H, c)` such that the
        contribution of the task to the QP objective of the IK is:

        .. math::

            \\| J \\Delta q - \\alpha e \\|_{W}^2 = \\frac{1}{2} \\Delta q^T H
            \\Delta q + c^T q

        The weight matrix :math:`W \\in \\mathbb{R}^{k \\times k}` weighs and
        normalizes task coordinates to the same unit. The unit of the overall
        contribution is [cost]^2. The configuration displacement :math:`\\Delta
        q` is the output of inverse kinematics (we divide it by :math:`\\Delta
        t` to get a commanded velocity).

        Args:
            robot: Robot model and its current configuration :math:`q`.

        Returns:
            Pair :math:`(H, c)` of Hessian matrix and linear vector of the QP
            objective.
        """
        raise NotImplementedError

    def __repr__(self):
        """
        Human-readable representation of the task.
        """
        return f"Task(gain={self.gain})"
