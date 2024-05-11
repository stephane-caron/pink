#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""
This module defines the BodySphericalBarrier class, which is a concrete implementation
of a barrier based on the distance between two frames.
"""

from typing import Iterable, Optional, Union

import numpy as np

from ..configuration import Configuration
from .barrier import Barrier


class BodySphericalBarrier(Barrier):
    r"""A barrier based on the distance between two frames.

    Defines a barrier function based on the Euclidean distance between two
    specified frames. It allows for the specification of a minimum distance
    threshold.

    The barrier function is defined as:

    .. math::

        h(\boldsymbol{q}) = \|\boldsymbol{p}_1(\boldsymbol{q}) - \boldsymbol{p}_2(\boldsymbol{q})\|^2 - d_{min}^2

    where :math:`\boldsymbol{p}_1(\boldsymbol{q})` and :math:`\boldsymbol{p}_2(\boldsymbol{q})` are the positions of the two frames
    in the world coordinate system, and :math:`d_{min}` is the minimum distance threshold.

    Attributes:
        frames: Tuple of two frame names.
        d_min: Minimum distance threshold.
    """

    frames: tuple[str, str]
    d_min: float

    def __init__(
        self,
        frame: tuple[str, str],
        d_min: float = None,
        gain: Union[float, np.ndarray] = 1.0,
        r: float = 3.0,
    ):
        """Initialize the BodySphericalBarrier.

        Args:
            frame: Tuple of two frame names.
            d_min: Minimum distance threshold.
            gain: Barrier gain. Defaults to 1.0.
            r: Weighting factor for the safe backup policy regularization term.
                Defaults to 3.0.
        """
        super().__init__(
            dim=1,
            gain=gain,
            # class_k_fn=lambda h: h / (1 + np.linalg.norm(h)),
            r=r,
        )
        self.frame = frame
        self.d_min = d_min

    def compute_barrier(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the value of the barrier function.

        The barrier function is computed based on the Euclidean distance between
        the two specified frames. It considers the minimum distance threshold.

        The barrier function is given by:

        .. math::

            h(\boldsymbol{q}) = \|\boldsymbol{p}_1(\boldsymbol{q}) - \boldsymbol{p}_2(\boldsymbol{q})\|^2 - d_{min}^2

        where :math:`\boldsymbol{p}_1(\boldsymbol{q})` and :math:`\boldsymbol{p}_2(\boldsymbol{q})` are the positions of the two frames
        in the world coordinate system, and :math:`d_{min}` is the minimum distance threshold.

        Args:
            configuration: Robot configuration :math:`\boldsymbol{q}`.

        Returns:
            Value of the barrier function :math:`\boldsymbol{h}(\boldsymbol{q})`.
        """
        pos1_world, pos2_world = self._get_frame_positions(configuration)
        return np.array([np.linalg.norm(pos1_world - pos2_world) ** 2 - self.d_min**2])

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the Jacobian matrix of the barrier function.

        The Jacobian matrix is computed based on the position Jacobians of the
        two specified frames. The Jacobians are transformed to align with the
        world coordinate system.

        The Jacobian matrix is given by:

        .. math::

            \frac{\partial h}{\partial \boldsymbol{q}}(\boldsymbol{q}) = 2(\boldsymbol{p}_1(\boldsymbol{q}) - \boldsymbol{p}_2(\boldsymbol{q}))^T \begin{bmatrix}
                \frac{\partial \boldsymbol{p}_1}{\partial \boldsymbol{q}}(\boldsymbol{q}) \\
                -\frac{\partial \boldsymbol{p}_2}{\partial \boldsymbol{q}}(\boldsymbol{q})
            \end{bmatrix}

        where :math:`\frac{\partial \boldsymbol{p}_1}{\partial \boldsymbol{q}}(\boldsymbol{q})` and :math:`\frac{\partial \boldsymbol{p}_2}{\partial \boldsymbol{q}}(\boldsymbol{q})`
        are the position Jacobians of the two frames.

        Args:
            configuration: Robot configuration :math:`\boldsymbol{q}`.

        Returns:
            Jacobian matrix :math:`\frac{\partial \boldsymbol{h}}{\partial \boldsymbol{q}}(\boldsymbol{q})`.
        """
        pos1_world, pos2_world = self._get_frame_positions(configuration)
        pos1_jac, pos2_jac = self._get_frame_jacobians(configuration)

        dh_dx = 2 * np.concatenate([pos1_world - pos2_world, pos2_world - pos1_world])
        dx_dq = np.vstack([pos1_jac, pos2_jac])
        return dh_dx.T @ dx_dq

    def _get_frame_positions(self, configuration: Configuration) -> tuple[np.ndarray, np.ndarray]:
        """Get the positions of the two frames in the world coordinate system.

        Args:
            configuration: Robot configuration.

        Returns:
            Tuple of position vectors of the two frames in the world coordinate system.
        """
        pos1_world = configuration.get_transform_frame_to_world(self.frame[0]).translation
        pos2_world = configuration.get_transform_frame_to_world(self.frame[1]).translation
        return pos1_world, pos2_world

    def _get_frame_jacobians(self, configuration: Configuration) -> tuple[np.ndarray, np.ndarray]:
        """Get the position Jacobians of the two frames in the world coordinate system.

        Args:
            configuration: Robot configuration.

        Returns:
            Tuple of position Jacobian matrices of the two frames in the world coordinate system.
        """
        pos1_jac = configuration.get_frame_jacobian(self.frame[0])[:3]
        rotation1 = configuration.get_transform_frame_to_world(self.frame[0]).rotation
        pos1_jac = rotation1 @ pos1_jac

        pos2_jac = configuration.get_frame_jacobian(self.frame[1])[:3]
        rotation2 = configuration.get_transform_frame_to_world(self.frame[1]).rotation
        pos2_jac = rotation2 @ pos2_jac

        return pos1_jac, pos2_jac