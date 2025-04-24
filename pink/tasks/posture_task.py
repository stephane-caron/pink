#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Posture task implementation."""

from typing import Optional

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from ..exceptions import TargetNotSet
from ..utils import get_root_joint_dim
from .task import Task


class PostureTask(Task):
    r"""Regulate joint angles to a desired posture.

    A posture is a vector of actuated joint angles. Floating base coordinates
    are not affected by this task.

    Attributes:
        target_q: Target vector in the configuration space. If the model has a
            floating base, then this vector should include floating-base
            coordinates (although they have no effect on the posture task).

    A posture task is typically used for regularization as it has a steady
    rank. For instance, when Upkie's legs are stretched and the Jacobian of its
    contact frames becomes singular, the posture task will drive the knees
    toward a preferred orientation.
    """

    target_q: Optional[np.ndarray]

    def __init__(
        self, cost: float, lm_damping: float = 0.0, gain: float = 1.0
    ) -> None:
        r"""Create task.

        Args:
            cost: value used to cast joint angle differences to a homogeneous
                cost, in :math:`[\mathrm{cost}] / [\mathrm{rad}]`.
            lm_damping: Unitless scale of the Levenberg-Marquardt (only when
                the error is large) regularization term, which helps when
                targets are unfeasible. Increase this value if the task is too
                jerky under unfeasible targets, but beware that too large a
                damping can slow down the task.
            gain: Task gain :math:`\alpha \in [0, 1]` for additional low-pass
                filtering. Defaults to 1.0 (no filtering) for dead-beat
                control.

        Note:
            We assume that the first seven coordinates of the configuration are
            for the floating base.
        """
        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)
        self.target_q = None

    def set_target(self, target_q: np.ndarray) -> None:
        """Set target posture.

        Args:
            target_q: Target vector in the configuration space. If the model
                has a floating base, then this vector should include
                floating-base coordinates (although they have no effect on the
                posture task).
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
        r"""Compute posture task error.

        The posture task error is defined as:

        .. math::

            e(q) = q^* \ominus q

        See :func:`pink.tasks.task.Task.compute_error` for more context.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Posture task error :math:`e(q)`.
        """
        if self.target_q is None:
            raise TargetNotSet("no posture target")
        _, root_nv = get_root_joint_dim(configuration.model)
        return pin.difference(
            configuration.model,
            self.target_q,
            configuration.q,
        )[root_nv:]

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the posture task Jacobian.

        The task Jacobian is the identity :math:`I_{n_v} \in \mathbb{R}^{n_v
        \times n_v}`, with :math:`n_v` the dimension of the robot's tangent
        space, so that the task dynamics are:

        .. math::

            J(q) \Delta q = \Delta q = \alpha (q^* \ominus q)

        See :func:`pink.tasks.task.Task.compute_jacobian` for more context.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Posture task Jacobian :math:`J(q)`.
        """
        _, nv = get_root_joint_dim(configuration.model)
        return configuration.tangent.eye[nv:, :]

    def __repr__(self):
        """Human-readable representation of the task."""
        return (
            "PostureTask("
            f"cost={self.cost}, "
            f"gain={self.gain}, "
            f"lm_damping={self.lm_damping})"
        )
