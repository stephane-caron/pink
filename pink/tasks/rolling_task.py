#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

"""Rolling task implementation."""

from typing import Optional, Sequence, Union

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from .task import Task


class RollingTask(Task):
    r"""Roll without slipping on a plane.

    Frames used in this task are:

    - Floor: inertial frame defining the contact plane the wheel rolls on.
    - Hub: frame attached to the hub of the wheel in the robot model.
    - Rim: its origin is the projection of the hub origin onto the floor
      plane. Its orientation is the same as the floor frame.

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

    hub_frame: str
    floor_frame: str
    wheel_radius: float

    def __init__(
        self,
        hub_frame: str,
        floor_frame: str,
        wheel_radius: float,
        cost: Optional[Union[float, Sequence[float], np.ndarray]],
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        r"""Define a new rolling task.

        Args:
            hub_frame: Name of a frame attached to the hub of the wheel in the
                robot model.
            floor_frame: Name of the inertial frame whose xy-plane defines the
                contact surface the wheel is rolling onto.
            wheel_radius: Radius of the wheel, i.e. distance in meters from the
                hub to the nearest point on the rim.
            cost: scalar or 3D cost vector,
                in :math:`[\mathrm{cost}] / [\mathrm{m}]`.
            lm_damping: Levenberg-Marquardt damping (see class attributes). The
                default value is conservatively low.
            gain: Task gain :math:`\alpha \in [0, 1]` for additional low-pass
                filtering. Defaults to 1.0 (no filtering) for dead-beat
                control.
        """
        super().__init__(
            cost=cost,
            gain=gain,
            lm_damping=lm_damping,
        )
        self.floor_frame = floor_frame
        self.hub_frame = hub_frame
        self.wheel_radius = wheel_radius

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the rolling task error.

        The error is a 3D vector :math:`{}_R e(q)` of the rim frame :math:`R`:

        .. math::

            {}_R e(q) := \begin{bmatrix}
                0 \\
                0 \\
                z_{\mathit{hub}} - \rho
                \end{bmatrix}

        The error is zero along the x- and y-axis of the rim frame to roll
        without slipping (*i.e.*, the velocity of the contact point is zero in
        the floor plane). The error along the z-axis of the rim frame is chosen
        so that the task keeps the wheel hub at distance :math:`\rho` (wheel
        radius) from the floor plane.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task error :math:`{}_R e(q) \in \mathbb{R}^3`, a translation
            expressed in the rim frame :math:`R`.
        """
        transform_hub_to_floor = configuration.get_transform(
            source=self.hub_frame, dest=self.floor_frame
        )
        position_hub_in_floor = transform_hub_to_floor.translation
        error_in_rim = np.zeros(3)
        error_in_rim[2] = position_hub_in_floor[2] - self.wheel_radius
        return error_in_rim

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the rolling task Jacobian.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Translation Jacobian matrix :math:`{}_R J_{WR}`, expressed locally
            in the rim frame :math:`R`.
        """
        transform_hub_to_floor = configuration.get_transform(
            source=self.hub_frame, dest=self.floor_frame
        )
        position_hub_in_floor = transform_hub_to_floor.translation
        position_rim_in_floor = position_hub_in_floor + np.array(
            [0.0, 0.0, -self.wheel_radius]
        )
        transform_rim_to_floor = pin.SE3(
            rotation=np.eye(3),
            translation=position_rim_in_floor,
        )
        transform_hub_to_rim = transform_rim_to_floor.actInv(
            transform_hub_to_floor
        )
        jacobian_hub_in_hub = configuration.get_frame_jacobian(self.hub_frame)
        jacobian_hub_in_rim = transform_hub_to_rim.action @ jacobian_hub_in_hub
        return jacobian_hub_in_rim[:3]  # translation

    def __repr__(self):
        """Human-readable representation of the task."""
        return (
            "RollingTask("
            f"hub_frame={self.hub_frame}, "
            f"floor_frame={self.floor_frame}, "
            f"wheel_radius={self.wheel_radius}, "
            f"cost={self.cost}, "
            f"gain={self.gain}, "
            f"lm_damping={self.lm_damping})"
        )
