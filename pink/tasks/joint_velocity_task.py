#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

"""Joint velocity task."""

from typing import Optional

import numpy as np

from ..configuration import Configuration
from ..utils import get_root_joint_dim
from .task import Task


class JointVelocityTask(Task):
    r"""Track joint velocities.

    This task minimizes :math:`\| v^{\mathit{ref}} - v \|_2` with :math:`v` the
    joint velocity resulting from differential IK and :math:`v^{\mathit{ref}}`
    is a reference joint-velocity vector.

    Note:
        Like the :class:`PostureTask`, this task does not apply to
        floating-base coordinates if the robot model has a floating base
        (a.k.a. root joint in Pinocchio).
    """

    target_v: Optional[np.ndarray]

    def __init__(
        self, cost: float, lm_damping: float = 0.0, gain: float = 1.0
    ) -> None:
        r"""Initialize task.

        Args:
            cost: Joint angular velocity cost, in
                :math:`[\mathrm{cost}] [\mathrm{s}] / [\mathrm{rad}]`.
        """
        super().__init__(
            cost=cost,
            gain=gain,
            lm_damping=lm_damping,
        )
        self.target_v = None

    def set_target(self, target_v: np.ndarray) -> None:
        """Set target joint velocity.

        Args:
            target_v: Target joint-velocity vector in the tangent space.
        """
        self.target_v = target_v.copy()

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the joint-velocity task error.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Joint-velocity task error.
        """
        _, root_nv = get_root_joint_dim(configuration.model)
        return np.zeros(configuration.model.nv - root_nv)

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the joint-velocity task Jacobian.

        The task Jacobian is the identity :math:`I_{n_v} \in \mathbb{R}^{n_v
        \times n_v}`, with :math:`n_v` the dimension of the robot's tangent
        space, so that the task dynamics are:

        .. math::

            J(q) \Delta q = \Delta q = \alpha (q^* \ominus q)

        See :func:`pink.tasks.task.Task.compute_jacobian` for more context.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Joint-velocity task Jacobian :math:`J(q)`.
        """
        _, nv = get_root_joint_dim(configuration.model)
        return configuration.tangent.eye[nv:, :]

    def __repr__(self):
        """Human-readable representation of the task."""
        return f"JointVelocityTask(cost={self.cost})"
