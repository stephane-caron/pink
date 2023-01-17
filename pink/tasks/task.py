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
All kinematic tasks derive from the :class:`Task` base class.

The formalism used in this implementation is written down in `this note on
task-based inverse kinematics
<https://scaron.info/robotics/inverse-kinematics.html>`_. As of February 2022
it hasn't been updated with the proper dimensional analysis, but the core
concepts and notations are there.
"""

import abc
from typing import Tuple

import numpy as np

from ..configuration import Configuration


class Task(abc.ABC):

    """
    Abstract base class for kinematic tasks.

    Attributes:
        gain: Task gain :math:`\\alpha \\in [0, 1]` for additional low-pass
            filtering. Defaults to 1.0 (no filtering) for dead-beat control.
    """

    gain: float = 1.0

    @abc.abstractmethod
    def compute_task_dynamics(
        self, configuration: Configuration
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
            configuration: Robot configuration to read values from.

        Returns:
            Tuple :math:`(J, e)` of Jacobian matrix and error vector, both
            expressed in the body frame.
        """

    @abc.abstractmethod
    def compute_qp_objective(
        self, configuration: Configuration
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
            configuration: Robot configuration to read values from.

        Returns:
            Pair :math:`(H, c)` of Hessian matrix and linear vector of the QP
            objective.
        """

    def __repr__(self):
        """Human-readable representation of the task."""
        return f"Task(gain={self.gain})"
