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
        model: pin.RobotWrapper,
        exclude_pairs: list[tuple[str, str]] | None = None,
        gain: Union[float, np.ndarray] = 1.0,
        safe_displacement_gain: float = 3.0,
    ):
        """..."""
        self.model = model
        self.model.collision_model.addAllCollisionPairs()

        # Removing specified collision pairs
        if exclude_pairs is not None:
            print("Number of collision pairs:", len(self.model.collision_model.collisionPairs))
            for frame1, frame2 in exclude_pairs:
                print(f"Removing collision pair {frame1} - {frame2}")
                self.model.collision_model.removeCollisionPair(
                    pin.CollisionPair(
                        self.model.collision_model.getGeometryId(frame1),
                        self.model.collision_model.getGeometryId(frame2),
                    )
                )
            print("Number of collision pairs:", len(self.model.collision_model.collisionPairs))
            # TODO: how to update collision data?
            self.model.collision_data = pin.GeometryData(self.model.collision_model)

        super().__init__(
            dim=len(self.model.collision_model.collisionPairs),
            gain=gain,
            safe_displacement_gain=safe_displacement_gain,
        )

        self.__q_prev = None

    def __update_collisions(self, configuration: Configuration):
        # Skip update if q is the same as the previous one.
        if self.__q_prev is not None and np.allclose(configuration.q, self.__q_prev):
            return
        self.__q_prev = configuration.q.copy()

        configuration.update(configuration.q)
        pin.computeCollisions(
            self.model.model,
            self.model.data,
            self.model.collision_model,
            self.model.collision_data,
            configuration.q,
            False,
        )
        pin.computeDistances(
            self.model.model,
            self.model.data,
            self.model.collision_model,
            self.model.collision_data,
            configuration.q,
        )
        pin.computeJointJacobians(self.model.model, self.model.data, configuration.q)

    def compute_barrier(self, configuration: Configuration) -> np.ndarray:
        """..."""
        self.__update_collisions(configuration)
        d = np.array(
            [
                self.model.collision_data.distanceResults[k].min_distance ** 2
                for k in range(len(self.model.collision_model.collisionPairs))
            ]
        )
        return d

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """..."""
        self.__update_collisions(configuration)

        J = np.zeros((self.dim, self.model.nq))
        Jrow_v = np.zeros(self.model.nv)

        for k in range(len(self.model.collision_model.collisionPairs)):
            cr = self.model.collision_data.collisionResults[k]
            cp = self.model.collision_model.collisionPairs[k]
            dr = self.model.collision_data.distanceResults[k]

            go_1 = self.model.collision_model.geometryObjects[cp.first]
            go_2 = self.model.collision_model.geometryObjects[cp.second]

            j1_id = go_1.parentJoint
            j2_id = go_2.parentJoint

            w1 = dr.getNearestPoint1()
            w2 = dr.getNearestPoint2()
            if cr.isCollision():
                print(f"Collision between {go_1.name} and {go_2.name}")
            r1 = w1 - self.model.data.oMi[j1_id].translation
            ddr_dw1 = 2 * dr.normal.reshape(1, -1)
            dw1_dj1 = np.block([np.eye(3), -pin.skew(r1)])
            dj1_dq = self.model.getJointJacobian(j1_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jrow_v = -ddr_dw1 @ dw1_dj1 @ dj1_dq

            r2 = w2 - self.model.data.oMi[j2_id].translation
            ddr_dw2 = 2 * dr.normal.reshape(1, -1)
            dw2_dj2 = np.block([np.eye(3), -pin.skew(r2)])
            dj2_dq = self.model.getJointJacobian(j2_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jrow_v += ddr_dw2 @ dw2_dj2 @ dj2_dq

            # if dr.min_distance < 0:
            #     Jrow_v = np.zeros(self.model.nv)
            J[k] = Jrow_v

        J = np.nan_to_num(J)

        return J
