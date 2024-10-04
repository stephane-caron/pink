#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

"""Rolling task implementation."""

import numpy as np
import pinocchio as pin
from pink import Configuration, Task


class RollingTask(Task):
    hub_frame: str
    floor_frame: str
    wheel_radius: float

    def __init__(
        self,
        hub_frame: str,
        floor_frame: str,
        wheel_radius: float,
        cost: float,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(
            cost=cost,
            gain=gain,
            lm_damping=lm_damping,
        )
        self.floor_frame = floor_frame
        self.hub_frame = hub_frame
        self.wheel_radius = wheel_radius

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        transform_hub_to_floor = configuration.get_transform(
            source=self.hub_frame, dest=self.floor_frame
        )
        position_hub_in_floor = transform_hub_to_floor.translation
        error_in_rim = np.zeros(3)
        error_in_rim[2] = position_hub_in_floor[2] - self.wheel_radius
        return error_in_rim

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
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
