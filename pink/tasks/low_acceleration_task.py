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
    :math:`v_{\mathrm{prev}}` is the latest integrated velocity. The Jacobian
    matrix is thus the identity, and the error opposite to
    :math:`v_{\mathrm{prev}}`.

    Attributes:
        v_prev: Latest integrated velocity.
    """

    v_prev: Optional[np.ndarray]

    def __init__(self, cost: float) -> None:
        r"""Initialize task.

        Args:
            cost: joint angular acceleration cost, in
                :math:`[\mathrm{cost}] [\mathrm{s}]^2 / [\mathrm{rad}]`.
        """
        super().__init__(cost=cost, gain=0.0, lm_damping=0.0)
        self.v_prev = None

    def set_previous_velocity(self, v_prev: np.ndarray) -> None:
        """Set the latest integrated velocity.

        The goal of the low-acceleration task is to minimize the difference
        between the new velocity and the previous one.

        Args:
            v_prev: Latest integrated velocity.
        """
        self.v_prev = v_prev.copy()

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the low-acceleration task error.

        The task error is opposite to the previous velocity, so that the
        overall task objective is :math:`\| v \ominus v_{\mathrm{prev}} \|^2`.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Posture task error :math:`e(q)`.
        """
        v_prev: np.ndarray = (
            self.v_prev
            if self.v_prev is not None
            else np.zeros(configuration.model.nv)
        )
        return -v_prev

    def __repr__(self):
        """Human-readable representation of the task."""
        return f"LowAccelerationTask(cost={self.cost})"
