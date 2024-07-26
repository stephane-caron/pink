#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

"""Damping task."""

import numpy as np

from ..configuration import Configuration
from .posture_task import PostureTask


class DampingTask(PostureTask):
    r"""Minimize joint velocities.

    The damping task minimizes :math:`\| v \|` with :math:`v` the velocity
    output of the differential IK. The word "damping" is used here by analogy
    with forces that fight against motion, and bring the robot to a rest if
    nothing else drives it.

    Note:
        The damping task is implemented as a special case of the posture task
        where the gain $\alpha$ is zero.
    """

    def __init__(self, cost: float) -> None:
        r"""Initialize task.

        Args:
            cost: joint angular velocity cost, in
                :math:`[\mathrm{cost}] [\mathrm{s}] / [\mathrm{rad}]`.
        """
        super().__init__(cost=cost, gain=0.0, lm_damping=0.0)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute damping task error.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Damping task error :math:`e(q) = 0`.
        """
        return np.zeros(configuration.model.nv)

    def __repr__(self):
        """Human-readable representation of the task."""
        return f"DampingTask(cost={self.cost})"
