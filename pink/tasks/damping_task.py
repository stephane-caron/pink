#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

"""Damping task."""

import numpy as np

from ..configuration import Configuration
from ..utils import get_root_joint_dim
from .posture_task import PostureTask


class DampingTask(PostureTask):
    r"""Minimize joint velocities.

    The damping task minimizes :math:`\| v \|_2` with :math:`v` the joint
    velocity resulting from differential IK. The word "damping" is used here by
    analogy with forces that fight against motion, and bring the robot to a
    rest if nothing else drives it.

    Note:
        The damping task is implemented as a special case of the
        :class:`PostureTask` where the gain $\alpha$ is zero.
    """

    def __init__(self, cost: float) -> None:
        r"""Initialize task.

        Args:
            cost: joint angular velocity cost, in
                :math:`[\mathrm{cost}] [\mathrm{s}] / [\mathrm{rad}]`.
        """
        super().__init__(
            cost=cost,
            gain=0.0,  # no gain: the task error is always zero
            lm_damping=0.0,  # no LM damping: the task error is always zero
        )

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute damping task error.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Damping task error :math:`e(q) = 0`.
        """
        _, root_nv = get_root_joint_dim(configuration.model)
        return np.zeros(configuration.model.nv - root_nv)

    def __repr__(self):
        """Human-readable representation of the task."""
        return f"DampingTask(cost={self.cost})"
