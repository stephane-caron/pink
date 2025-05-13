#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

"""Omniwheel task implementation."""

import numpy as np

from ..configuration import Configuration
from .rolling_task import RollingTask


class OmniwheelTask(RollingTask):
    r"""Roll without slipping on a plane, allowing lateral motion.

    The name of this task comes from omniwheels, also known as poly wheels.
    See the :class:`pink.tasks.rolling_task.RollingTask`.

    Attributes:
        hub_frame: Name of a frame attached to the hub of the wheel in the
            robot model.
        floor_frame: Name of the inertial frame whose xy-plane defines the
            contact surface the wheel is rolling onto.
        wheel_radius: Radius of the wheel, i.e. distance in meters from the hub
            to the nearest point on the rim.

    Note:
        See also the similar `wheel task in the PlaCo C++ library
        <https://github.com/Rhoban/placo/blob/9e65abadff45071ab11274eb74770d71e10b7ca8/docs/kinematics/wheel_task.rst>`__.
    """

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the rolling task error.

        The error is a 2D vector :math:`{}_R e(q)` of the rim frame :math:`R`:

        .. math::

            {}_R e(q) := \begin{bmatrix}
                0 \\
                z_{\mathit{hub}} - \rho
                \end{bmatrix}

        The error is zero along the x-axis of the rim frame to roll without
        slipping (*i.e.*, the velocity of the contact point is zero in the
        floor plane). The error along the z-axis of the rim frame is chosen so
        that the task keeps the wheel hub at distance :math:`\rho` (wheel
        radius) from the floor plane.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task error :math:`{}_R e(q) \in \mathbb{R}^3`, a translation
            expressed in the rim frame :math:`R`.
        """
        rolling_error = super().compute_error(configuration)
        return np.array([rolling_error[0], rolling_error[2]])

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the omniwheel task Jacobian.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Jacobian matrix :math:`{}_R J_{WR}`, expressed locally in the rim
            frame :math:`R`, of shape :math:`(2, n_q)`.
        """
        rolling_jacobian = super().compute_jacobian(configuration)
        return np.vstack([rolling_jacobian[0], rolling_jacobian[2]])

    def __repr__(self):
        """Human-readable representation of the task."""
        rolling_repr = super().__repr__()
        return rolling_repr.replace("RollingTask", "OmniwheelTask")
