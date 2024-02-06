#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

"""Damping task."""

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from ..utils import get_root_joint_dim
from .posture_task import PostureTask


class DampingTask(PostureTask):
    r"""Minimize joint velocities.

    The word "damping" is used here by analogy with forces that fight against
    motion, and bring the robot to a rest if nothing else drives it.

    The damping task is implemented as a special case of the posture task where
    the gain $\alpha$ is zero.
    """

    def __init__(self, cost: float, lm_damping: float = 0.0) -> None:
        r"""Initialize task.

        Args:
            cost: joint angular velocity cost, in
                :math:`[\mathrm{cost}] [\mathrm{s}] / [\mathrm{rad}]`.
            lm_damping: Unitless scale of the Levenberg-Marquardt (only when
                the error is large) regularization term, which helps when
                targets are unfeasible. Increase this value if the task is too
                jerky under unfeasible targets, but beware that too large a
                damping can slow down the task.
        """
        super().__init__(cost=cost, gain=0.0, lm_damping=lm_damping)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute damping task error.

        The damping task error is defined as:

        .. math::

            e(q) = q_0 \ominus q,

        where :math:`q_0` is the neutral configuration of the robot. The
        damping task error is not relevant in itself, as the gain $\alpha$ of a
        damping task is always zero. (Yet we still calculate it properly so
        that the Jacobian is the derivative of the error, as unit tested.) See
        :func:`Task.compute_error` for more context.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Posture task error :math:`e(q)`.
        """
        _, nv = get_root_joint_dim(configuration.model)
        return pin.difference(
            configuration.model,
            configuration.q,
            pin.neutral(configuration.model),
        )[nv:]

    def __repr__(self):
        """Human-readable representation of the task."""
        return (
            "DampingTask("
            f"cost={self.cost}, "
            f"lm_damping={self.lm_damping})"
        )
