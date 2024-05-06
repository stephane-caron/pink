#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""General description"""

from typing import Iterable, Optional, Union

import numpy as np

from ..configuration import Configuration
from .barrier import CBF


class BodySphericalCBF(CBF):
    r"""Abstract class description.

    Attributes:
        ...
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
        """..."""
        super().__init__(
            dim=1,
            gain=gain,
            # class_k_fn=lambda h: h / (1 + np.linalg.norm(h)),
            r=r,
        )
        self.frame = frame
        self.d_min = d_min

    def compute_barrier(self, configuration: Configuration) -> np.ndarray:
        """..."""
        pos1_world = configuration.get_transform_frame_to_world(self.frame[0]).translation
        pos2_world = configuration.get_transform_frame_to_world(self.frame[1]).translation

        return np.array([np.linalg.norm(pos1_world - pos2_world) ** 2 - self.d_min**2])

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """..."""
        pos1_world = configuration.get_transform_frame_to_world(self.frame[0]).translation
        pos2_world = configuration.get_transform_frame_to_world(self.frame[1]).translation

        pos1_jac = configuration.get_frame_jacobian(self.frame[0])[:3]
        # Transform jacobian to world aligned frame
        rotation1 = configuration.get_transform_frame_to_world(self.frame[0]).rotation
        pos1_jac = rotation1 @ pos1_jac

        pos2_jac = configuration.get_frame_jacobian(self.frame[1])[:3]
        # Transform jacobian to world aligned frame
        rotation2 = configuration.get_transform_frame_to_world(self.frame[1]).rotation
        pos2_jac = rotation2 @ pos2_jac

        dh_dx = 2 * np.concatenate([pos1_world - pos2_world, pos2_world - pos1_world])
        dx_dq = np.vstack([pos1_jac, pos2_jac])
        return dh_dx.T @ dx_dq
