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
Add fields to a pinocchio.Model that are useful for IK.
"""

from typing import List

import numpy as np
import pinocchio as pin


class Tangent:

    """
    Represent the tangent space at a given robot configuration.

    Attributes:
        bounded_proj: Projection matrix on bounded joints.
        eye: Identity matrix.
        ones: Tangent space vector with all coordinates at one.
        zeros: Zero tangent space vector.
    """

    bounded_proj: np.ndarray
    eye: np.ndarray
    ones: np.ndarray
    zeros: np.ndarray

    def __init__(self, model: pin.Model, bounded_tangent_idx):
        eye = np.eye(model.nv)
        ones = np.ones(model.nv)
        zeros = np.zeros(model.nv)
        bounded_proj = eye[bounded_tangent_idx]
        self.bounded_proj = bounded_proj
        self.eye = eye
        self.ones = ones
        self.zeros = zeros


def list_bounded_joints(model: pin.Model) -> List[pin.JointModel]:
    """
    List bounded joints in a pinocchio model.

    Args:
        model: robot model.

    Returns:
        List of bounded joints.
    """
    has_configuration_limit = np.logical_and(
        model.upperPositionLimit < 1e20,
        model.upperPositionLimit > model.lowerPositionLimit + 1e-10,
    )
    return [
        joint
        for joint in model.joints
        if has_configuration_limit[
            slice(joint.idx_q, joint.idx_q + joint.nq)
        ].all()
    ]


def extend_pinocchio_model(model: pin.Model) -> None:
    """
    Extend model with configuration-independent quantities.

    Args:
        model: robot model to extend.

    Returns:
        Array of bounded joint indexes.
    """
    if hasattr(model, "bounded_joints"):
        return

    bounded_joints = list_bounded_joints(model)
    bounded_config_idx = []
    bounded_tangent_idx = []
    for joint in bounded_joints:
        bounded_config_idx.extend(range(joint.idx_q, joint.idx_q + joint.nq))
        bounded_tangent_idx.extend(range(joint.idx_v, joint.idx_v + joint.nv))
    bounded_config_idx = np.array(bounded_config_idx)
    bounded_tangent_idx = np.array(bounded_tangent_idx)
    bounded_config_idx.setflags(write=False)
    bounded_tangent_idx.setflags(write=False)
    bounded_config_eye = np.eye(model.nq)[bounded_config_idx]
    bounded_tangent_eye = np.eye(model.nv)[bounded_tangent_idx]

    model.bounded_config_eye = bounded_config_eye
    model.bounded_config_idx = bounded_config_idx
    model.bounded_joints = bounded_joints
    model.bounded_tangent_eye = bounded_tangent_eye
    model.bounded_tangent_idx = bounded_tangent_idx
    model.bounded_velocity_limit = model.velocityLimit[bounded_tangent_idx]
    model.nv_bounded = len(bounded_joints)
    model.tangent = Tangent(model, bounded_tangent_idx)
