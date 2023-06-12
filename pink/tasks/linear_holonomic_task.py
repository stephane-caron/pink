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

r"""Linear holonomic task :math:`A (q \ominus q_0) = b`."""

from typing import Sequence, Tuple, Union

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from .task import Task


class LinearHolonomicTask(Task):
    r"""Linear holonomic task :math:`A (q \ominus q_0) = b`.

    Attributes:
        A: matrix that defines the task:

            .. raw:: latex html

                \begin{align}
                e(q) & = A (q \ominus^\ell q_0) - b \\
                \dot{e}(q) & := A \dot{q}
                \end{align}

            where :math: `e(q) \in \mathbb{R}^{k}` is the quantity that the
            task aims to drive to zero (:math:`k` is the dimension of the
            task).
        b: vector that defines the affine part of the task
        q0: element for which we work in the Lie algebra. If set to None, it
            will use the neutral configuration of the robot.
        cost: joint angular error cost in
            :math:`[\mathrm{cost}] / [\mathrm{rad}]`.

    Note:
        A linear holonomic task is typically used for a robot
        that has mechanical constraint (e.g., closed loop kinematics).
        Floating base coordinates are not affected by this task.

    Notes:
        To be fully explicit, our quantities in the task equation :math:`e(q) =
        A (q \ominus^\ell q_0) - b` belong to the following sets:

            .. raw:: latex html

                \begin{align}
                e(q) & \in \mathbb{R}^p &
                A & \in \mathcal{L}(\mathfrak{g}^\ell_{q_0}, \mathbb{R}^p) &
                b & \in \mathbb{R}^p
                \end{align}

        where :math:`\mathfrak{g}^\ell_{q_0}` is the Lie algebra associated
        with the Lie group :math:`\mathcal{C}` (our configuration space), taken
        in the local frame (:math:`\ell`, *a.k.a* body frame) at the reference
        configuration :math:`q_0`.

        We take the *local* difference :math:`q \ominus^\ell q_0` between our
        configurations :math:`q \in \mathcal{C}` and :math:`q_0 \in
        \mathcal{C}`, which is an element of the local Lie algebra in
        :math:`\mathfrak{g}^\ell_{q_0}` at :math:`q_0`. In Pinocchio the
        convention is to use local differences, unless specified otherwise
        (e.g. world-aligned frames). It should not be confused with the general
        Lie difference :math:`\ominus` (that one is never used in practice),
        nor the spatial difference :math:`\ominus^s`, which is an element of
        the spatial Lie algebra :math:`\mathfrak{g}^s_{q_0}` associated with
        :math:`q_0`:

            .. raw:: latex html

                \begin{align}
                q \ominus q_0 & \in \mathcal{T}_{q_0} \mathcal{C} \\
                q \ominus^\ell q_0 & \in \mathfrak{g}^\ell_{q_0} \\
                q \ominus^s q_0 & \in \mathfrak{g}^s_{q_0}
                \end{align}

        On a side note, here's a look at the dimensions of these elements:

            .. raw:: latex html

                \begin{align}
                \mathrm{dim}(q) = \mathrm{dim}(q_0) = n_q \\
                \mathrm{dim}(q \ominus q_0) = n_q \\
                \mathrm{dim}(q \ominus^\ell q_0)
                = \mathrm{dim}(q \ominus^s q_0)
                = n_v
                \end{align}

        Finally, we derive the Jacobian of our error :math:`e(q)` as follows,
        denoting by :math:`a = \log(q_0^{-1} q)` so that :math:`q \ominus^\ell
        q_0 = \log(a)`:

            .. raw:: latex html

                \begin{align}
                \left.\frac{\partial e(q)}{\partial q} \right|_{\ell}
                = A \left. \frac{\partial \log}{\partial q}
                \right|_{\ell}^{\ell} \left.
                \frac{\partial a}{\partial q} \right|_{\ell} = A \left.
                \frac{\partial \log}{\partial q}
                \right|_{\ell}
                \end{align}

        By virtue of the identity :math:`\left.{}^\ell \frac{\partial
        a}{\partial q}\right|_{\ell} = I_{\mathfrak{g}}`. (See Yann de
        Mont-Marin's future blog for details :p)
    """

    A: np.ndarray
    cost: Union[float, Sequence[float]]

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        q_0: np.ndarray,
        cost: Union[float, Sequence[float]],
    ) -> None:
        r"""Create task.

        Args:
            A: Jacobian matrix of the task, of size :math:`p \times n_v`.
            b: target vector of the task, of dimension :math:`p`.
            q_0: stationary configuration :math:`q_0 \in \mathcal{C}`
                associated with the task, of dimension :math:`n_q`.
            cost: cost vector of dimension :math:`p`.
        """
        assert A.shape[0] == 1 if isinstance(cost, float) else len(cost)
        self.A = A
        self.b = b
        self.cost = cost
        self.q_0 = q_0

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the task error function.

        The error function :math:`e(q) \in \mathbb{R}^{p}` is the quantity that
        the task aims to drive to zero (:math:`p` is the dimension of the
        task). It appears in the first-order task dynamics:

        .. math::

            J(q) \Delta q = \alpha e(q)

        See the description of the task above for details about the calculation
        of this Jacobian.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task error vector :math:`e(q)`.
        """
        q_ref = (
            pin.neutral(configuration.model)
            if self.q0 is None else self.q0
        )
        return (
            self.A
            @ pin.difference(configuration.model, q_ref, configuration.q)
            - self.b
        )

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the task Jacobian at a given configuration.

        The task Jacobian :math:`J(q) \in \mathbb{R}^{p \times n_v}` appears in
        the first-order task dynamics:

        .. math::

            J(q) \Delta q = \alpha e(q)

        See the description of the task above for details about the calculation
        of this Jacobian.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task Jacobian :math:`J(q)`.
        """
        q_ref = (
            pin.neutral(configuration.model)
            if self.q0 is None else self.q0
        )
        return self.A @ pin.dDifference(
            configuration.model, q_ref, self.q, pin.ARG1
        )

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
        gain_error = self.gain * self.compute_error(configuration)
        weight = np.diag(
            [self.cost] * jacobian.shape[0]
            if isinstance(self.cost, float)
            else self.cost
        )
        weighted_jacobian = np.dot(weight, jacobian)  # [cost]
        weighted_error = np.dot(weight, gain_error)  # [cost]
        H = weighted_jacobian.T @ weighted_jacobian
        c = -weighted_error.T @ weighted_jacobian
        return (H, c)

    def __repr__(self):
        """Human-readable representation of the task."""
        return f"LinearHolonomicTask(cost={self.cost}, gain={self.gain})"
