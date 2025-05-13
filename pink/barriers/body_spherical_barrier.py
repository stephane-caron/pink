#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Ivan Domrachev, Simeon Nedelchev

"""Barrier based on the distance between two frames."""

from typing import Tuple, Union

import numpy as np

from ..configuration import Configuration
from ..exceptions import NegativeMinimumDistance
from .barrier import Barrier


class BodySphericalBarrier(Barrier):
    r"""A barrier based on the distance between two frames.

    Defines a barrier function based on the Euclidean distance between two
    specified frames. It allows for the specification of a minimum distance
    threshold.

    The barrier function is defined as:

    .. math::

        h(q) = \|p_1(q) -
        p_2(q)\|^2 - d_{min}^2

    where :math:`p_1(q)` and
    :math:`p_2(q)` are the positions of the two
    frames in the world coordinate system, and :math:`d_{min}`
    is the minimum distance threshold.

    Attributes:
        frames: Tuple of two frame names.
        d_min: Minimum distance threshold.
    """

    frames: Tuple[str, str]
    d_min: float

    def __init__(
        self,
        frames: Tuple[str, str],
        d_min: float,
        gain: Union[float, np.ndarray] = 1.0,
        safe_displacement_gain: float = 3.0,
    ):
        """Initialize the BodySphericalBarrier.

        Args:
            frames: Tuple of two frame names.
            d_min: Minimum distance threshold.
            gain: Barrier gain. Defaults to 1.0.
            safe_displacement_gain: gain for the safe backup displacement
                cost term. Defaults to 3.0.
        """
        if d_min < 0.0:
            raise NegativeMinimumDistance(
                "The minimum distance threshold must be non-negative."
            )
        super().__init__(
            dim=1,
            gain=gain,
            gain_function=lambda h: h / (1 + np.abs(h)),
            safe_displacement_gain=safe_displacement_gain,
        )
        self.frames = frames
        self.d_min = d_min

    def compute_barrier(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the value of the barrier function.

        The barrier function is computed based on the Euclidean distance
        between the two specified frames. It considers the minimum distance
        threshold.

        The barrier function is given by:

        .. math::

            h(q) = \|p_1(q) -
            p_2(q)\|^2 - d_{min}^2

        where :math:`p_1(q)` and
        :math:`p_2(q)` are the positions of the two
        frames in the world coordinate system, and :math:`d_{min}`
        is the minimum distance threshold.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Value of the barrier function
                :math:`h(q)`.
        """
        pos1_world, pos2_world = self._get_frame_positions(configuration)
        return np.array(
            [
                (pos1_world - pos2_world).dot(pos1_world - pos2_world)
                - self.d_min**2
            ]
        )

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the Jacobian matrix of the barrier function.

        The Jacobian matrix is computed based on the position Jacobians of the
        two specified frames. The Jacobians are transformed to align with the
        world coordinate system.

        The Jacobian matrix is given by:

        .. math::

            \frac{\partial h}{\partial q}(q) =
            2(p_1(q) -
            p_2(q))^T \begin{bmatrix}
                \frac{\partial p_1}{\partial q}
                (q) \\
                -\frac{\partial p_2}{\partial q}
                (q)
            \end{bmatrix}

        where :math:`\frac{\partial p_1}{\partial q}(q)`
        and :math:`\frac{\partial p_2}{\partial q}(q)`
        are the position Jacobians of the two frames.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Jacobian matrix
                :math:`\frac{\partial h}{\partial q}(q)`.
        """
        pos1_world, pos2_world = self._get_frame_positions(configuration)
        pos1_jac, pos2_jac = self._get_frame_jacobians(configuration)

        dh_dx = 2 * (pos1_world - pos2_world)
        dx_dq = pos1_jac - pos2_jac
        return dh_dx.T @ dx_dq

    def _get_frame_positions(
        self, configuration: Configuration
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the positions of the two frames in the world coordinate system.

        Args:
            configuration: Robot configuration.

        Returns:
            Tuple of position vectors of the two frames in the world
                coordinate system.
        """
        pos1_world = configuration.get_transform_frame_to_world(
            self.frames[0]
        ).translation
        pos2_world = configuration.get_transform_frame_to_world(
            self.frames[1]
        ).translation
        return pos1_world, pos2_world

    def _get_frame_jacobians(
        self, configuration: Configuration
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the position Jacobians of the two frames in the world frame.

        Args:
            configuration: Robot configuration.

        Returns:
            Tuple of position Jacobian matrices of the two frames in the world
                coordinate system.
        """
        pos1_jac = configuration.get_frame_jacobian(self.frames[0])[:3]
        rotation1 = configuration.get_transform_frame_to_world(
            self.frames[0]
        ).rotation
        pos1_jac = rotation1 @ pos1_jac

        pos2_jac = configuration.get_frame_jacobian(self.frames[1])[:3]
        rotation2 = configuration.get_transform_frame_to_world(
            self.frames[1]
        ).rotation
        pos2_jac = rotation2 @ pos2_jac

        return pos1_jac, pos2_jac
