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

"""Linear Holonomic task implementation."""

from typing import Sequence, Tuple, Union

import numpy as np

from ..configuration import Configuration
from .exceptions import TaskJacobianNotSet
from .task import Task


class LinearHolonomicTask(Task):
    r"""General class for linear holonomic tasks.

    Attributes:
        A: matrix that relates the following relationship:

        .. math::
            e(q) = Aq - b
            \dot{e}(q) := A\dot{q}

        where :math: `e(q) \in \mathbb{R}^{k}` is the quantity that the task 
            aims to derive to zero (:math:`k` is the dimension of the task).
        cost: joint angular error cost in 
            :math:`[\mathrm{cost}] / [\mathrm{rad}]`.

    Note:
        A linear holonomic task is typically used for a robot 
        that has mechanical constraint (e.g., closed loop kinematics).
        Floating base coordinates are not affected by this task.
    """

    A: np.ndarray
    cost: Union[float, Sequence[float]]

    def __init__(
        self, A: np.ndarray, cost: Union[float, Sequence[float]]
    ) -> None:
        r"""Create task.

        Args:
            A: Jacobian matrix of a linear holonomic contraint.
            cost: joint angular error cost in
                :math:`[\mathrm{cost}] / [\mathrm{rad}]`.
        """
        if isinstance(cost, float):
            assert A.shape[0] == 1
        else:
            assert A.shape[0] == len(cost)

        self.A = A
        self.cost = cost

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the task error function.

        The error function :math:`e(q) \in \mathbb{R}^{k}` is the quantity that
        the task aims to drive to zero (:math:`k` is the dimension of the
        task). It appears in the first-order task dynamics:

        .. math::

            J(q) \Delta q = \alpha e(q)

        The Jacobian matrix :math:`J(q) \in \mathbb{R}^{k \times n_v}`,
        with :math:`n_v` the dimension of the robot's tangent space, is
        computed by :func:`Task.compute_jacobian`, while the configuration
        displacement :math:`\\Delta q` is the output of inverse kinematics. The
        error vector :math:`e(q)` is multiplied by the task gain :math:`\alpha
        \in [0, 1]`. The gain is usually 1 for dead-beat control (*i.e.*
        converge as fast as possible), but it can also be lower for some extra
        low-pass filtering.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task error vector :math:`e(q)`.
        """
        if self.A is None:
            raise TaskJacobianNotSet("no task Jacobian set")

        return np.zeros(self.A.shape[0])

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the task Jacobian at a given configuration.

        The task Jacobian :math:`J(q) \in \mathbb{R}^{k \times n_v}` appears in
        the first-order task dynamics:

        .. math::

            J(q) \Delta q = \alpha e(q)

        The error :math:`e(q) \in \mathbb{R}^{k \times n_v}`, with :math:`k`
        the dimension of the task and :math:`n_v` the dimension of the robot's
        tangent space, is computed by :func:`Task.compute_error`, while the
        configuration displacement :math:`\\Delta q` is the output of inverse
        kinematics.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task Jacobian :math:`J(q)`.
        """
        if self.A is None:
            raise TaskJacobianNotSet("no task Jacobian set")

        return self.A

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
            Pair :math:`(H)` of Hessian matrix of the QP objective.
        """
        jacobian = self.compute_jacobian(configuration)
        gain_error = self.gain * self.compute_error(
            configuration
        )  # np.zeros(k)
        weight = (
            self.cost if isinstance(self.cost, float) else np.diag(self.cost)
        )
        weighted_jacobian = np.dot(weight, jacobian)  # [cost]
        weighted_error = np.dot(weight, gain_error)  # [cost] np.zeros(k)
        H = weighted_jacobian.T @ weighted_jacobian
        c = -weighted_error.T @ weighted_jacobian  # np.zeros(H.shape[0])
        return (H, c)

    def __repr__(self):
        """Human-readable representation of the task."""
        return f"LinearHolonomicTask(cost={self.cost}, gain={self.gain})"
