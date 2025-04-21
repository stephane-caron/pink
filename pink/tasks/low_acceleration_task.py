#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Low acceleration task."""

from typing import Optional

import numpy as np

from ..configuration import Configuration
from .posture_task import PostureTask


class LowAccelerationTask(PostureTask):
    r"""Minimize joint accelerations.

    The task is to minimize :math:`\| v \ominus v_{\mathrm{prev}} \|` where
    :math:`v_{\mathrm{prev}}` is the latest integrated velocity.

    Attributes:
        Delta_q_prev: Latest displacement of the robot.

    Note:
        This task does not dissipate energy and will thus tend to yield
        oscillations. Considering coupling it with a :class:`DampingTask`.
    """

    Delta_q_prev: Optional[np.ndarray]

    def __init__(self, cost: float) -> None:
        r"""Initialize task.

        Args:
            cost: joint angular acceleration cost, in
                :math:`[\mathrm{cost}] [\mathrm{s}]^2 / [\mathrm{rad}]`.
        """
        super().__init__(
            cost=cost,
            gain=1.0,  # this task hacks around the definition of error
            lm_damping=0.0,  # idem
        )
        self.Delta_q_prev = None

    def set_last_integration(self, v_prev: np.ndarray, dt) -> None:
        """Set the latest velocity and the duration it was applied for.

        The goal of the low-acceleration task is to minimize the difference
        between the new velocity and the previous one.

        Args:
            v_prev: Latest integrated velocity.
            dt: Integration timestep in [s].
        """
        self.Delta_q_prev = v_prev * dt

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the low-acceleration task error.

        The task error is opposite to the previous velocity, so that the
        overall task objective is :math:`\| v \ominus v_{\mathrm{prev}} \|^2`.

        Args:
            configuration: Robot configuration :math:`q`.

        Return:
            Low-acceleration task error
            :math:`e(q) = -\mathrm{d}t v_{\mathrm{prev}}`. This choice, along
            with :math:`\alpha = 1` and the Jacobian :math:`J` being the
            identity, ensures that :math:`J(q) \Delta q = -\alpha e(q)
            \Leftrightarrow v = \Delta q / \mathrm{d}t = v_{\mathrm{prev}}`.

        """
        Delta_q_prev: np.ndarray = (
            self.Delta_q_prev
            if self.Delta_q_prev is not None
            else np.zeros(configuration.model.nv)
        )
        return -Delta_q_prev

    def __repr__(self):
        """Human-readable representation of the task."""
        return f"LowAccelerationTask(cost={self.cost})"
