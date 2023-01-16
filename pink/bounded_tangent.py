#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Stéphane Caron
# Copyright 2023 Inria
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

from typing import List, Optional

import numpy as np
import pinocchio as pin

from .utils import VectorSpace


class BoundedTangent(VectorSpace):

    """
    Subspace of the tangent space restricted to bounded joints.

    Attributes:
        nv: Dimension of the full tangent space.
    """

    indices: np.ndarray
    joints: list
    nv: int
    projection_matrix: Optional[np.ndarray]
    velocity_limit: Optional[np.ndarray]

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
            if joint.idx_q >= 0
            and has_configuration_limit[
                slice(joint.idx_q, joint.idx_q + joint.nq)
            ].all()
        ]

        index_list: List[int] = []
        for joint in joints:
            index_list.extend(range(joint.idx_v, joint.idx_v + joint.nv))
        indices = np.array(index_list)
        indices.setflags(write=False)

        dim = len(indices)
        super().__init__(dim)
        projection_matrix = np.eye(model.nv)[indices] if dim > 0 else None

        self.indices = indices
        self.joints = joints
        self.nv = model.nv
        self.projection_matrix = projection_matrix
        self.velocity_limit = (
            model.velocityLimit[indices] if len(joints) > 0 else None
        )

    def project(self, v: np.ndarray) -> np.ndarray:
        """
        Project a vector from the tangent space to the bounded tangent
        subspace.

        Args:
            v: Vector from the original space.
        """
        assert v.shape == (self.nv,), "Dimension mismatch"
        return v[self.indices]
