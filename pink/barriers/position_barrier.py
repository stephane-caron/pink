#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Frame position barrier."""

from typing import List, Optional, Union

import numpy as np

from ..configuration import Configuration
from .barrier import Barrier
from .exceptions import NoPositionLimitProvided


class PositionBarrier(Barrier):
    r"""A position-based barrier.

    Defines a barrier function based on the position of a
    specified frame in the world coordinate system. It allows for the
    specification of minimum and maximum position bounds along selected axes.

    Attributes:
        frame: Name of the frame to monitor.
        indices: Indices of the position components to consider.
        p_min: Minimum position bounds.
        p_max: Maximum position bounds.
    """

    frame: str
    indices: List[int]
    p_min: Optional[np.ndarray]
    p_max: Optional[np.ndarray]

    def __init__(
        self,
        frame: str,
        indices: Optional[List[int]] = None,
        p_min: Optional[np.ndarray] = None,
        p_max: Optional[np.ndarray] = None,
        gain: Union[float, np.ndarray] = 1.0,
        r: float = 0.0,
    ):
        """Initialize a position barrier.

        Args:
            frame (str): name of the frame to monitor.
            indices (Optional[List[int]], optional): indices with limits.
                Indices from 0 to 2 stand for coordinates x to z respectively.
                Defaults to all dimensions.
            p_min (Optional[np.ndarray], optional): minimum position limit.
                Defaults to None, meaning no minimum limit is applied.
            p_max (Optional[np.ndarray], optional): maximum position limit.
                Defaults to None, meaning no maximum limit is applied.
            gain (Union[float, np.ndarray], optional): regularization gain.
                Defaults to 1.0.
            r (float, optional): penalty weighting parameter. Defaults to 0.0.

        Raises:
            NoPositionLimitProvided: neither minimum nor maximum
                position limit is provided.
        """
        indices = [0, 1, 2] if indices is None else indices

        if p_min is None and p_max is None:
            raise NoPositionLimitProvided(
                f"Position barrier for frame {frame} requires either \
                    p_min or p_max"
            )

        dim = 0
        if p_min is not None:
            dim += len(indices)
        if p_max is not None:
            dim += len(indices)

        # If both p_min and p_max are specified, they must have
        # the same gains repeated along the dimension
        if isinstance(gain, np.ndarray) and len(gain) != dim:
            gain = np.tile(gain, 2)

        super().__init__(
            dim,
            gain=gain,
            r=r,
        )

        self.indices = indices
        self.frame = frame
        self.p_min = p_min
        self.p_max = p_max

    def compute_barrier(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the value of the barrier function.

        The barrier function is computed based on the position of the specified
        frame in the world coordinate system. It considers the minimum and
        maximum position bounds along the selected axes.

        Args:
            configuration: Robot configuration :math:`\boldsymbol{q}`.

        Returns:
            Value of the barrier function
            :math:`\boldsymbol{h}(\boldsymbol{q})`.
        """
        pos_world = configuration.get_transform_frame_to_world(
            self.frame
        ).translation
        barriers = []
        if self.p_min is not None:
            barriers.append(pos_world[self.indices] - self.p_min)
        if self.p_max is not None:
            barriers.append(self.p_max - pos_world[self.indices])

        return np.concatenate(barriers)

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the Jacobian matrix of the barrier function.

        The Jacobian matrix is computed based on the position Jacobian of the
        specified frame. The Jacobian is transformed to align with the world
        coordinate system and only the selected indices are considered.

        Args:
            configuration: Robot configuration :math:`\boldsymbol{q}`.

        Returns:
            Jacobian matrix
            :math:`\frac{\partial \boldsymbol{h}}{\partial \boldsymbol{q}}(\boldsymbol{q})`.
        """  # noqa: E501
        pos_jac = configuration.get_frame_jacobian(self.frame)[:3]
        # Transform jacobian to world aligned frame
        rotation = configuration.get_transform_frame_to_world(
            self.frame
        ).rotation
        pos_jac = rotation @ pos_jac

        # Select only relevant indices
        pos_jac = pos_jac[self.indices]

        jacobians = []
        if self.p_min is not None:
            jacobians.append(pos_jac.copy())
        if self.p_max is not None:
            jacobians.append(-pos_jac.copy())

        return np.vstack(jacobians)
