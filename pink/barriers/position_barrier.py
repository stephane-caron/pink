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
        r: float = 3.0,
    ):
        """..."""
        indices = [0, 1, 2] if indices is None else indices

        dim = 0
        if p_min is not None:
            dim += len(indices)
        if p_max is not None:
            dim += len(indices)

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

        return np.hstack(jacobians)