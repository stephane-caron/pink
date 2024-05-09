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


class SelfCollisionBarrier(CBF):
    r"""Abstract class description.

    Attributes:
        ...
    """

    frames: tuple[str, str]
    d_min: float

    def __init__(
        self,
        model: pin.RobotWrapper,
        exclude_pairs: list[tuple[str, str]] = [],
        gain: Union[float, np.ndarray] = 1.0,
        r: float = 3.0,
    ):
        """..."""
        self.model = model
        self.model.collision_model.addAllCollisionPairs()
        for frame1, frame2 in exclude_pairs:
            print(frame1, frame2)
            self.model.collision_model.removeCollisionPair(
                pin.CollisionPair(
                    self.model.model.getFrameId(frame1),
                    self.model.model.getFrameId(frame2),
                )
            )
            self.model.collision_data.deactivateCollisionPair(
                self.model.model.getFrameId(frame1),
                self.model.model.getFrameId(frame2),
            )

        super().__init__(
            dim=len(self.model.collision_model.collisionPairs),
            gain=gain,
            # class_k_fn=lambda h: h / (1 + np.linalg.norm(h)),
            r=r,
        )

        self.__q_prev = None

    def __update_collisions(self, configuration: Configuration):
        if self.__q_prev is not None and np.allclose(configuration.q, self.__q_prev):
            return
        self.__q_prev = configuration.q.copy()

        pin.computeCollisions(
            self.model.model,
            self.model.data,
            self.model.collision_model,
            self.model.collision_data,
            configuration.q,
            False,
        )
        print("Two")
        pin.computeDistances(
            self.model.model,
            self.model.data,
            self.model.collision_model,
            self.model.collision_data,
            configuration.q,
        )
        print("hi!")
        pin.updateGeometryPlacements(
            self.model.model,
            self.model.data,
            self.model.collision_model,
            self.model.collision_data,
            configuration.q,
        )
        print("Here?")

    def compute_barrier(self, configuration: Configuration) -> np.ndarray:
        """..."""
        self.__update_collisions(configuration)
        print("Here??")
        return np.array(
            [
                self.model.collision_data.distanceResults[k].min_distance
                for k in range(len(self.model.collision_model.collisionPairs))
            ]
        )

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """..."""
        self.__update_collisions(configuration)

        J = np.zeros((self.dim, self.model.nq))
        Jrow_v = np.zeros(self.model.nv)

        for k in range(len(self.model.collision_model.collisionPairs)):
            cp = self.model.collision_model.collisionPairs[k]
            dr = self.model.collision_data.distanceResults[k]

            go_1 = self.model.collision_model.geometryObjects[cp.first]
            go_2 = self.model.collision_model.geometryObjects[cp.second]

            j1_id = go_1.parentJoint
            j2_id = go_2.parentJoint
            if j1_id - j2_id == 1:
                continue
            w1 = dr.getNearestPoint1()
            r1 = w1 - self.model.data.oMi[j1_id].translation
            ddr_dw1 = -dr.normal.reshape(1, -1)
            dw1_dj1 = np.block([np.eye(3), -pin.skew(r1)])
            dj1_dq = pin.getJointJacobian(
                self.model.model, self.model.data, j1_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            Jrow_v = ddr_dw1 @ dw1_dj1 @ dj1_dq

            w2 = dr.getNearestPoint1()
            r2 = w2 - self.model.data.oMi[j2_id].translation
            ddr_dw2 = dr.normal.reshape(1, -1)
            dw2_dj2 = np.block([np.eye(3), -pin.skew(r2)])
            dj2_dq = pin.getJointJacobian(
                self.model.model, self.model.data, j1_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            Jrow_v = ddr_dw2 @ dw2_dj2 @ dj2_dq

            J[k] = Jrow_v
        print("Here????")
        return J
