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

"""All kinematic tasks derive from the :class:`Task` base class.

The formalism used in this implementation is written down in `this note on
task-based inverse kinematics
<https://scaron.info/robotics/inverse-kinematics.html>`_. As of February 2022
it hasn't been updated with the proper dimensional analysis, but the core
concepts and notations are there.
"""

import abc
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ..configuration import Configuration


class Task(abc.ABC):
    r"""Abstract base class for kinematic tasks.

    Attributes:
        cost: cost vector with the same dimension as the error of the task. Its
            units depends on the error as well.
        gain: Task gain :math:`\alpha \in [0, 1]` for additional low-pass
            filtering. Defaults to 1.0 (no filtering) for dead-beat control.
        lm_damping: Unitless scale of the Levenberg-Marquardt (only when
            the error is large) regularization term, which helps when
            targets are unfeasible. Increase this value if the task is too
            jerky under unfeasible targets, but beware that a larger
            damping slows down the task.
    """

    cost: Optional[Union[float, Sequence[float], np.ndarray]]
    gain: float
    lm_damping: float

    def __init__(
        self,
        cost: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        r"""Create task.

        Args:
            cost: cost vector with the same dimension as the error of the task.
                Its units depends on the error as well.
            gain: Task gain :math:`\alpha \in [0, 1]` for additional low-pass
                filtering. Defaults to 1.0 (no filtering) for dead-beat
                control.
            lm_damping: Unitless scale of the Levenberg-Marquardt (only when
                the error is large) regularization term, which helps when
                targets are unfeasible. Increase this value if the task is too
                jerky under unfeasible targets, but beware that too large a
                damping can slow down the task.
        """
        self.cost = cost
        self.gain = gain
        self.lm_damping = lm_damping

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    def compute_qp_objective(
        self, configuration: Configuration
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the matrix-vector pair :math:`(H, c)` of the QP objective.

        This pair is such that the contribution of the task to the QP objective
        of the IK is:

        .. math::

            \| J \Delta q - \alpha e \|_{W}^2 = \frac{1}{2} \Delta q^T H
            \Delta q + c^T q

        The weight matrix :math:`W \in \mathbb{R}^{k \times k}` weighs and
        normalizes task coordinates to the same unit. The unit of the overall
        contribution is [cost]^2. The configuration displacement :math:`\Delta
        q` is the output of inverse kinematics (we divide it by :math:`\Delta
        t` to get a commanded velocity).

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Pair :math:`(H(q), c(q))` of Hessian matrix and linear vector of
            the QP objective.

        See Also:
            Levenberg-Marquardt damping is described in [Sugihara2011]_. The
            dimensional analysis in this class is our own.
        """
        jacobian = self.compute_jacobian(configuration)
        gain_error = self.gain * self.compute_error(configuration)

        weight = (
            np.eye(jacobian.shape[0])
            if self.cost is None
            else np.diag(
                [self.cost] * jacobian.shape[0]
                if isinstance(self.cost, float)
                else self.cost
            )
        )

        weighted_jacobian = weight @ jacobian  # [cost]
        weighted_error = weight @ gain_error  # [cost]
        mu = self.lm_damping * weighted_error @ weighted_error  # [cost]^2
        eye_tg = configuration.tangent.eye
        # Our Levenberg-Marquardt damping `mu * eye_tg` is isotropic in the
        # robot's tangent space. If it helps we can add a tangent-space scaling
        # to damp the floating base differently from joint angular velocities.
        H = weighted_jacobian.T @ weighted_jacobian + mu * eye_tg
        c = -weighted_error.T @ weighted_jacobian
        return (H, c)

    def __repr__(self):
        """Human-readable representation of the task."""
        return (
            f"Task("
            f"cost={self.cost}, "
            f"gain={self.gain}, "
            f"lm_damping={self.lm_damping})"
        )
