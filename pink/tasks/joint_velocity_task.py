#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

"""Joint velocity task."""

import numpy as np

from ..configuration import Configuration
from ..exceptions import TaskDefinitionError, TargetNotSet
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

    def __init__(self, cost: float) -> None:
        r"""Initialize task.

        Args:
            cost: Joint angular velocity cost, in
                :math:`[\mathrm{cost}] [\mathrm{s}] / [\mathrm{rad}]`.
        """
        super().__init__(
            cost=cost,
            gain=0.0,  # no gain: the task error is directly a velocity
            lm_damping=0.0,  # no LM damping either
        )
        self.__target_Delta_q = None

    def set_target(self, target_v: np.ndarray, dt: float) -> None:
        """Set target joint velocity.

        Args:
            target_v: Target joint-velocity vector in the tangent space.
        """
        self.__target_Delta_q = target_v.copy()

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the joint-velocity task error.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Joint-velocity task error.
        """
        _, root_nv = get_root_joint_dim(configuration.model)
        task_nv = configuration.model.nv - root_nv
        if self.__target_Delta_q is None:
            raise TargetNotSet(repr(self))
        elif self.__target_Delta_q.shape[0] != task_nv:
            raise TaskDefinitionError(
                f"Target has dimension nv={self.__target_Delta_q.shape[0]} "
                f"but the task expects nv={task_nv} "
                f"({configuration.model.nv=}, {root_nv=})"
            )
        return self.__target_Delta_q

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the joint-velocity task Jacobian.

        The task Jacobian is the identity :math:`I_{n_v} \in \mathbb{R}^{n_v
        \times n_v}`, with :math:`n_v` the dimension of the robot's tangent
        space, so that the task dynamics are:

        .. math::

            J(q) \Delta q = \Delta q

        Combining this with :math:`e(q) = \Delta q_{ref}` and :math:`\alpha =
        1`, we get an overall velocity-tracking task dynamics:

        .. math::

            J(q) \Delta q = -\alpha e(q)
            \quad \Leftrightarrow \quad
            \Delta q = \Delta q_{ref}

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Joint-velocity task Jacobian :math:`J(q)`.
        """
        _, root_nv = get_root_joint_dim(configuration.model)
        return configuration.tangent.eye[root_nv:, :]

    def __repr__(self):
        """Human-readable representation of the task."""
        return f"JointVelocityTask(cost={self.cost})"
