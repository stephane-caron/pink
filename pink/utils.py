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
Utility classes and functions.
"""

from typing import Tuple

import numpy as np
import pinocchio as pin


def custom_configuration_vector(robot: pin.Model, **kwargs) -> np.ndarray:
    """
    Generate a configuration vector where named joints have specific values.

    Args:
        robot: Robot model.

    Returns:
        Configuration vector where named joints have the values specified in
        keyword arguments, and other joints have their neutral value.
    """
    q = pin.neutral(robot.model)
    for joint_name, joint_value in kwargs.items():
        joint_id = robot.model.getJointId(joint_name)
        joint = robot.model.joints[joint_id]
        q[joint.idx_q] = joint_value
    return q


def get_root_joint_dim(model: pin.Model) -> Tuple[int, int]:
    """
    Count the configuration and tangent dimensions of the root joint, if any.

    Args:
        model: Robot model.

    Returns:
        nq: Number of configuration dimensions.
        nv: Number of tangent dimensions.
    """
    if model.existJointName("root_joint"):
        root_joint_id = model.getJointId("root_joint")
        root_joint = model.joints[root_joint_id]
        return root_joint.nq, root_joint.nv
    return 0, 0


class VectorSpace:

    """
    Wrapper to refer to a vector space and its characteristic matrices.

    Attributes:
        dim: Dimension of the vector space.
        eye: Identity matrix.
        ones: Vector full of ones, dimension of the space.
        zeros: Zero vector of the space.
    """

    dim: int
    eye: np.ndarray
    ones: np.ndarray
    zeros: np.ndarray

    def __init__(self, dim: int):
        eye = np.eye(dim)
        ones = np.ones(dim)
        zeros = np.zeros(dim)
        eye.setflags(write=False)
        ones.setflags(write=False)
        zeros.setflags(write=False)
        self.dim = dim
        self.eye = eye
        self.ones = ones
        self.zeros = zeros
