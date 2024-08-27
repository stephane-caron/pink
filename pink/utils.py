#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Utility classes and functions."""

from typing import Tuple

import numpy as np
import pinocchio as pin

from .exceptions import ConfigurationError, PinkError


def custom_configuration_vector(robot: pin.Model, **kwargs) -> np.ndarray:
    """Generate a configuration vector where named joints have specific values.

    Args:
        robot: Robot model.
        kwargs: Custom values for joint coordinates.

    Returns:
        Configuration vector where named joints have the values specified in
        keyword arguments, and other joints have their neutral value.
    """
    q = pin.neutral(robot.model)
    for name, value in kwargs.items():
        joint_id = robot.model.getJointId(name)
        joint = robot.model.joints[joint_id]
        value = np.array(value).flatten()
        if value.shape[0] != joint.nq:
            raise ConfigurationError(
                f"Joint '{name}' has {joint.nq=} but is set to {value.shape=}"
            )
        q[joint.idx_q : joint.idx_q + joint.nq] = value
    return q


def get_root_joint_dim(model: pin.Model) -> Tuple[int, int]:
    """Count configuration and tangent dimensions of the root joint, if any.

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


def get_joint_idx(model: pin.Model, joint_name: str) -> Tuple[int, int]:
    """Get joint index in the configuration and tangent space.

    Args:
        model: Robot model.
        joint_name: Joint name.

    Returns:
        idx_q: Joint idx in configuration space.
        idx_v: Joint idx in tangent space.
    """
    if model.existJointName(joint_name):
        joint_id = model.getJointId(joint_name)
        joint = model.joints[joint_id]
        return joint.idx_q, joint.idx_v
    raise PinkError(
        f"cannot find the joint index corresponding to joint {joint_name}"
    )


class VectorSpace:
    """Wrapper to refer to a vector space and its characteristic matrices."""

    __eye: np.ndarray
    __ones: np.ndarray
    __zeros: np.ndarray

    def __init__(self, dim: int):
        """Create new vector space description.

        Args:
            dim: Dimension.
        """
        eye = np.eye(dim)
        ones = np.ones(dim)
        zeros = np.zeros(dim)
        eye.setflags(write=False)
        ones.setflags(write=False)
        zeros.setflags(write=False)
        self.__eye = eye
        self.__ones = ones
        self.__zeros = zeros

    @property
    def eye(self) -> np.ndarray:
        """Identity matrix from and to the vector space."""
        return self.__eye

    @property
    def ones(self) -> np.ndarray:
        """Vector full of ones, dimension of the space."""
        return self.__ones

    @property
    def zeros(self) -> np.ndarray:
        """Zero vector of the space."""
        return self.__zeros


def process_collision_pairs(
    model: pin.Model, collision_model: pin.GeometryModel, srdf_path: str = ""
) -> pin.GeometryData:
    """Process collision pairs.

    Args:
        model: robot model.
        collision_model: Collision model of the robot.
        srdf_path: Path to the SRDF file, which used to exclude collision
            pairs. Defaults to empty string, meaning no collision pairs are
            excluded.

    Returns:
        collision_data: Collision data, generated after updating
            collision_model.
    """
    collision_model.addAllCollisionPairs()
    if srdf_path != "":
        pin.removeCollisionPairs(model, collision_model, srdf_path)

    # Collision models have been modified => re-generate corresponding data.
    collision_data = pin.GeometryData(collision_model)

    # Enable contact detection for avoiding Nans at collisions
    collision_data.enable_contact = True

    return collision_data
