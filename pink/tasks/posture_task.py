#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Stéphane Caron
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Posture task specification.
"""

from typing import Optional, Tuple

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from ..utils import get_root_joint_dim
from .exceptions import TargetNotSet
from .task import Task


class PostureTask(Task):

    """
    Regulate joint angles to a desired posture, *i.e.* a vector of actuated
    joint angles. Floating base coordinates are not affected by this task.

    Attributes:
        cost: joint angular error cost in
            :math:`[\\mathrm{cost}] / [\\mathrm{rad}]`.
        target_q: Target vector in the configuration space.

    A posture task is typically used for regularization as it has a steady
    rank. For instance, when Upkie's legs are stretched and the Jacobian of its
    contact frames become singular, the posture task will drive the knees
    toward a preferred orientation.
    """

    cost: float
    target_q: Optional[np.ndarray]

    def __init__(self, cost: float) -> None:
        """
        Create task.

        Args:
            cost: joint angular error cost in
                :math:`[\\mathrm{cost}] / [\\mathrm{rad}]`.

        Note:
            We assume that the first seven coordinates of the configuration are
            for the floating base.
        """
        self.cost = cost
        self.target_q = None

    def set_target(self, target_q: np.ndarray) -> None:
        """
        Set target posture.

        Args:
            target_q: Target vector in the configuration space.
        """
        self.target_q = target_q.copy()

    def set_target_from_configuration(
        self, configuration: Configuration
    ) -> None:
        """
        Set target posture from a robot configuration.

        Args:
            configuration: Robot configuration.
        """
        self.set_target(configuration.q)

    def compute_task_dynamics(
        self, configuration: Configuration
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the matrix :math:`J(q)` and vector :math:`\\alpha e(q)` such
        that the task dynamics are:

        .. math::

            J(q) \\Delta q = \\alpha e(q)

        The Jacobian matrix is :math:`J(q) \\in \\mathbb{R}^{n \\times n}`,
        with :math:`n` the dimension of the robot's tangent space, and the
        error vector is :math:`e(q) \\in \\mathbb{R}^n`. Both depend on the
        configuration :math:`q` of the robot.

        See :func:`Task.compute_task_dynamics` for more context.

        Args:
            configuration: Robot configuration to read kinematics from.

        Returns:
            Pair :math:`(J, \\alpha e)` of Jacobian matrix and error vector,
            both expressed in the body frame.
        """
        if self.target_q is None:
            raise TargetNotSet("no posture target")
        _, nv = get_root_joint_dim(configuration.model)
        jacobian = configuration.tangent.eye[nv:, :]
        error = pin.difference(
            configuration.model, configuration.q, self.target_q
        )[nv:]
        return (jacobian, self.gain * error)

    def compute_qp_objective(
        self, configuration: Configuration
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Hessian matrix :math:`H` and linear vector :math:`c` such
        that the contribution of the task to the QP objective is:

        .. math::

            \\| J \\Delta q - \\alpha e \\|_{W}^2
            = \\frac{1}{2} \\Delta q^T H \\Delta q + c^T q

        The weight matrix :math:`W \\in \\mathbb{R}^{n \\times n}` weighs and
        normalizes task coordinates to the same unit. The unit of the overall
        contribution is :math:`[\\mathrm{cost}]^2`. The configuration
        displacement :math:`\\Delta q` is the output of inverse kinematics (we
        divide it by :math:`\\Delta t` to get a commanded velocity).

        Args:
            robot: Robot model and configuration.

        Returns:
            Pair :math:`(H, c)` of Hessian matrix and linear vector of the QP
            objective.
        """
        jacobian, error = self.compute_task_dynamics(configuration)
        weighted_jacobian = self.cost * jacobian  # [cost]
        weighted_error = self.cost * error  # [cost]
        H = weighted_jacobian.T @ weighted_jacobian
        c = -weighted_error.T @ weighted_jacobian
        return (H, c)

    def __repr__(self):
        """
        Human-readable representation of the task.
        """
        return f"PostureTask(cost={self.cost}, gain={self.gain})"
