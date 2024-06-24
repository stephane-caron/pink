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
        safe_displacement_gain: float = 1.0,
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
        return np.array(
            [
                # configuration.collision_data.distanceResults[k].min_distance - self.d_min
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

            f1_id = go_1.parentJoint
            f2_id = go_2.parentJoint

            w1 = dr.getNearestPoint1()
            w2 = dr.getNearestPoint2()

            r1 = w1 - data.oMi[f1_id].translation
            r2 = w2 - data.oMi[f2_id].translation
            # TODO: normal for some reason it not calculated
            n = (w1 - w2) / np.linalg.norm(w1 - w2)

            # Approach 1: from casadi_kin_dyn
            J_1 = pin.getJointJacobian(model, data, f1_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jrow_v = n.T @ J_1[:3, :] + (np.cross(r1, n)).T @ J_1[3:, :]

            J_2 = pin.getJointJacobian(model, data, f2_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jrow_v -= n.T @ J_2[:3, :] + (np.cross(r2, n)).T @ J_2[3:, :]

            # Approach 2: mine derivations
            # ddr_dw1 = 2 * n.reshape(1, -1)
            # dw1_dj1 = np.block([np.eye(3), pin.skew(r1)])
            # dj1_dq = pin.getJointJacobian(model, data, f1_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            # Jrow_v = ddr_dw1 @ dw1_dj1 @ dj1_dq

            # ddr_dw2 = 2 * n.reshape(1, -1)
            # dw2_dj2 = np.block([np.eye(3), pin.skew(r2)])
            # dj2_dq = pin.getJointJacobian(model, data, f2_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            # Jrow_v -= ddr_dw2 @ dw2_dj2 @ dj2_dq

            # if dr.min_distance < 0:
            #     Jrow_v = np.zeros(self.model.nv)
            J[k] = Jrow_v.copy()

        J = np.nan_to_num(J)

        return J
