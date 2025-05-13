#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Ivan Domrachev, Simeon Nedelchev

"""Frame position barrier."""

from typing import List, Optional, Union

import numpy as np

from ..configuration import Configuration
from ..exceptions import NoPositionLimitProvided
from .barrier import Barrier


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
        safe_displacement_gain: float = 0.0,
    ):
        """Initialize a position barrier.

        Args:
            frame: name of the frame to monitor.
            indices: indices with limits.
                Indices from 0 to 2 stand for coordinates x to z respectively.
                Defaults to all dimensions.
            p_min: minimum position limit.
                Defaults to None, meaning no minimum limit is applied.
            p_max: maximum position limit.
                Defaults to None, meaning no maximum limit is applied.
            gain: regularization gain.
                Defaults to 1.0.
            safe_displacement_gain: gain for the safe backup displacement
                cost term. Defaults to 0.0.

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
            safe_displacement_gain=safe_displacement_gain,
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
            configuration: Robot configuration :math:`q`.

        Returns:
            Value of the barrier function
            :math:`h(q)`.
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
            configuration: Robot configuration :math:`q`.

        Returns:
            Jacobian matrix
            :math:`\frac{\partial h}{\partial q}(q)`.
        """
        pos_jacobian_in_frame = configuration.get_frame_jacobian(self.frame)[
            :3
        ]
        # Transform jacobian to world aligned frame
        rotation = configuration.get_transform_frame_to_world(
            self.frame
        ).rotation
        # Apply rotation to trasfrom from world aligned to local frame
        # and select only relevant indices
        pos_jacobian_in_world = (rotation @ pos_jacobian_in_frame)[
            self.indices
        ]

        jacobians = []
        if self.p_min is not None:
            jacobians.append(pos_jacobian_in_world.copy())
        if self.p_max is not None:
            jacobians.append(-pos_jacobian_in_world.copy())

        return np.vstack(jacobians)
