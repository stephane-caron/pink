#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""General description"""
from typing import Union

import numpy as np
import pinocchio as pin

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
        frames: tuple[str, str],
        O1: np.ndarray | None = None,
        O2: np.ndarray | None = None,
        d_min: float = 0,
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
        self.frame = frames
        self.O1 = O1 if O1 is not None else np.zeros(3)
        self.O2 = O2 if O2 is not None else np.zeros(3)
        self.d_min = d_min

    def compute_barrier(self, configuration: Configuration) -> np.ndarray:
        """..."""
        pos1_world = configuration.get_transform_frame_to_world(self.frame[0]).translation + self.O1
        pos2_world = configuration.get_transform_frame_to_world(self.frame[1]).translation + self.O2

        return np.array([np.linalg.norm(pos1_world - pos2_world) ** 2 - self.d_min**2])

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """..."""
        pos1_world = configuration.get_transform_frame_to_world(self.frame[0]).translation + self.O1
        pos2_world = configuration.get_transform_frame_to_world(self.frame[1]).translation + self.O2

        pos1_jac = configuration.get_frame_jacobian(self.frame[0])
        # Transform jacobian to world aligned frame
        rotation1 = configuration.get_transform_frame_to_world(self.frame[0]).rotation
        pos1_jac[:3] = rotation1 @ pos1_jac[:3]
        pos1_jac = np.block([np.eye(3), -pin.skew(self.O1)]) @ pos1_jac

        pos2_jac = configuration.get_frame_jacobian(self.frame[1])
        # Transform jacobian to world aligned frame
        rotation2 = configuration.get_transform_frame_to_world(self.frame[1]).rotation
        pos2_jac[:3] = rotation2 @ pos2_jac[:3]
        pos2_jac = np.block([np.eye(3), -pin.skew(self.O2)]) @ pos2_jac

        dh_dx = 2 * np.concatenate([pos1_world - pos2_world, pos2_world - pos1_world])
        dx_dq = np.vstack([pos1_jac, pos2_jac])
        return dh_dx.T @ dx_dq
