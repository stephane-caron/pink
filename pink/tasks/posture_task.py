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

"""Posture task implementation."""

from typing import Optional, Tuple

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from ..utils import get_root_joint_dim
from .exceptions import TargetNotSet
from .task import Task


class PostureTask(Task):
    r"""Regulate joint angles to a desired posture.

    A posture is a vector of actuated joint angles. Floating base coordinates
    are not affected by this task.

    Attributes:
        cost: joint angular error cost in
            :math:`[\mathrm{cost}] / [\mathrm{rad}]`.
        target_q: Target vector in the configuration space.

    A posture task is typically used for regularization as it has a steady
    rank. For instance, when Upkie's legs are stretched and the Jacobian of its
    contact frames become singular, the posture task will drive the knees
    toward a preferred orientation.
    """

    cost: float
    target_q: Optional[np.ndarray]

    def __init__(self, cost: float) -> None:
        r"""Create task.

        Args:
            cost: joint angular error cost in
                :math:`[\mathrm{cost}] / [\mathrm{rad}]`.

        Note:
            We assume that the first seven coordinates of the configuration are
            for the floating base.
        """
        self.cost = cost
        self.target_q = None

    def set_target(self, target_q: np.ndarray) -> None:
        """Set target posture.

        Args:
            target_q: Target vector in the configuration space.
        """
        self.target_q = target_q.copy()

    def set_target_from_configuration(
        self, configuration: Configuration
    ) -> None:
        """Set target posture from a robot configuration.

        Args:
            configuration: Robot configuration.
        """
        self.set_target(configuration.q)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Compute posture task error.

        See :func:`Task.compute_error` for more context.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Posture task error :math:`e(q)`.
        """
        if self.target_q is None:
            raise TargetNotSet("no posture target")
        _, nv = get_root_joint_dim(configuration.model)
        return pin.difference(
            configuration.model, configuration.q, self.target_q
        )[nv:]

    def compute_jacobian(
        self, configuration: Configuration
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute posture task Jacobian.

        The task Jacobian is the identity :math:`I_{n_v} \in \mathbb{R}^{n_v
        \times n_v}`, with :math:`n_v` the dimension of the robot's tangent
        space, so that the task dynamics are:

        .. math::

            J(q) \Delta q = \Delta_q = \alpha (q^* - q)

        See :func:`Task.compute_jacobian` for more context.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Posture task Jacobian :math:`J(q)`.
        """
        if self.target_q is None:
            raise TargetNotSet("no posture target")
        _, nv = get_root_joint_dim(configuration.model)
        return configuration.tangent.eye[nv:, :]

    def compute_qp_objective(
        self, configuration: Configuration
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the matrix-vector pair :math:`(H, c)` of the QP objective.

        This pair is such that the contribution of the task to the QP objective
        of the IK is:

        .. math::

            \| J \Delta q - \alpha e \|_{W}^2
            = \frac{1}{2} \Delta q^T H \Delta q + c^T q

        The weight matrix :math:`W \in \mathbb{R}^{n \times n}` weighs and
        normalizes task coordinates to the same unit. The unit of the overall
        contribution is :math:`[\mathrm{cost}]^2`. The configuration
        displacement :math:`\Delta q` is the output of inverse kinematics (we
        divide it by :math:`\Delta t` to get a commanded velocity).

        Args:
            configuration: Robot configuration.

        Returns:
            Pair :math:`(H, c)` of Hessian matrix and linear vector of the QP
            objective.
        """
        jacobian = self.compute_jacobian(configuration)
        gain_error = self.gain * self.compute_error(configuration)
        weighted_jacobian = self.cost * jacobian  # [cost]
        weighted_error = self.cost * gain_error  # [cost]
        H = weighted_jacobian.T @ weighted_jacobian
        c = -weighted_error.T @ weighted_jacobian
        return (H, c)

    def __repr__(self):
        """Human-readable representation of the task."""
        return f"PostureTask(cost={self.cost}, gain={self.gain})"
