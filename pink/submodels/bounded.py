#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Stéphane Caron
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Subset of bounded joints associated with a robot model.
"""

from dataclasses import dataclass

import numpy as np
import pinocchio as pin


@dataclass
class Projection:

    """
    Projection rom one space (configuration or tangent) to its subspace when
    restricting joints to bounded ones.

    Attributes:
        input_dim: Dimension of input space.
        indices: Coordinates of bounded joints in subspace.
        projection_matrix: Projection matrix from the input space to that space
            restricted to bounded joints.
    """

    input_dim: int
    indices: np.ndarray
    projection_matrix: np.ndarray

    def project(self, v: np.ndarray) -> np.ndarray:
        """
        Project a vector from the original space to the subspace.

        Args:
            v: Vector from the original space.
        """
        assert v.shape == (self.input_dim,), "Dimension mismatch"
        return v[self.indices]


class Bounded:
    def __init__(self, model: pin.Model):
        """
        Bounded joints in a robot model.

        Args:
            model: robot model.

        Returns:
            List of bounded joints.
        """
        has_configuration_limit = np.logical_and(
            model.upperPositionLimit < 1e20,
            model.upperPositionLimit > model.lowerPositionLimit + 1e-10,
        )

        joints = [
            joint
            for joint in model.joints
            if has_configuration_limit[
                slice(joint.idx_q, joint.idx_q + joint.nq)
            ].all()
        ]

        config_idx = []
        tangent_idx = []
        for joint in joints:
            config_idx.extend(range(joint.idx_q, joint.idx_q + joint.nq))
            tangent_idx.extend(range(joint.idx_v, joint.idx_v + joint.nv))
        config_idx = np.array(config_idx)
        tangent_idx = np.array(tangent_idx)
        config_idx.setflags(write=False)
        tangent_idx.setflags(write=False)
        config_proj = np.eye(model.nq)[config_idx]
        tangent_proj = np.eye(model.nv)[tangent_idx]

        self.configuration = Projection(model.nq, config_idx, config_proj)
        self.joints = joints
        self.nv = len(joints)
        self.tangent = Projection(model.nv, tangent_idx, tangent_proj)
        self.velocity_limit = model.velocityLimit[tangent_idx]
