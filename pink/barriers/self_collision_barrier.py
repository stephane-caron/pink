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
from .barrier import Barrier


class SelfCollisionBarrier(Barrier):
    r"""Abstract class description.

    Attributes:
        ...
    """

    frames: tuple[str, str]
    d_min: float

    def __init__(
        self,
        n_collision_pairs: int,
        gain: Union[float, np.ndarray] = 1.0,
        safe_displacement_gain: float = 3.0,
        d_min: float = 0.02,
    ):
        """..."""
        super().__init__(
            dim=n_collision_pairs,
            gain=gain,
            safe_displacement_gain=safe_displacement_gain,
        )

        self.d_min = d_min
        self.__q_prev = None

    def compute_barrier(self, configuration: Configuration) -> np.ndarray:
        """..."""
        print("BARRIER VALUE: ")
        print(
            np.array(
                [
                    configuration.collision_data.distanceResults[k].min_distance ** 2 - self.d_min**2
                    for k in range(len(configuration.collision_model.collisionPairs))
                ]
            )
        )
        return np.array(
            [
                configuration.collision_data.distanceResults[k].min_distance ** 2 - self.d_min**2
                for k in range(len(configuration.collision_model.collisionPairs))
            ]
        )

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """..."""
        model = configuration.model
        data = configuration.data
        collision_model = configuration.collision_model
        collision_data = configuration.collision_data

        J = np.zeros((self.dim, model.nq))
        print(J.shape)

        for k in range(len(collision_model.collisionPairs)):
            cr = collision_data.collisionResults[k]
            cp = collision_model.collisionPairs[k]
            dr = collision_data.distanceResults[k]

            go_1 = collision_model.geometryObjects[cp.first]
            go_2 = collision_model.geometryObjects[cp.second]

            j1_id = go_1.parentFrame
            j2_id = go_2.parentFrame

            w1 = dr.getNearestPoint1()
            w2 = dr.getNearestPoint2()

            if cr.isCollision():
                print(f"Collision between {go_1.name} and {go_2.name}")
            r1 = w1 - data.oMf[j1_id].translation
            ddr_dw1 = dr.normal.reshape(1, -1)
            dw1_dj1 = np.block([np.eye(3), -pin.skew(r1)])
            dj1_dq = pin.getFrameJacobian(model, data, j1_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jrow_v = -ddr_dw1 @ dw1_dj1 @ dj1_dq

            r2 = w2 - data.oMf[j2_id].translation
            ddr_dw2 = dr.normal.reshape(1, -1)
            dw2_dj2 = np.block([np.eye(3), -pin.skew(r2)])
            dj2_dq = pin.getFrameJacobian(model, data, j2_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jrow_v += ddr_dw2 @ dw2_dj2 @ dj2_dq

            # if dr.min_distance < 0:
            #     Jrow_v = np.zeros(self.model.nv)
            J[k] = Jrow_v.copy()

        J = np.nan_to_num(J)
        print("JACOBIAN:")
        print(J)

        return J
