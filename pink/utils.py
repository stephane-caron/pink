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
Utility functions.
"""

import os.path
from importlib import import_module  # type: ignore
from typing import List, Union

import numpy as np
import pinocchio as pin

try:
    import robot_descriptions
except ImportError:
    robot_descriptions = None


PinocchioJoint = Union[
    pin.JointModelRX,
    pin.JointModelRY,
    pin.JointModelRZ,
    pin.JointModelPX,
    pin.JointModelPY,
    pin.JointModelPZ,
    pin.JointModelFreeFlyer,
    pin.JointModelSpherical,
    pin.JointModelSphericalZYX,
    pin.JointModelPlanar,
    pin.JointModelTranslation,
]


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


def load_urdf(
    urdf_path: str,
    package_dirs: List[str] = [],
    root_joint_type: PinocchioJoint = pin.JointModelFreeFlyer,
) -> pin.RobotWrapper:
    """
    Load the URDF of a robot description.

    Args:
        urdf_path: Path to the URDF file.
        package_dirs: List of directories Pinocchio can use as guesses to
            expand relative and ``package://`` paths in the URDF.
        root_joint_type: Type of the first joint of the kinematic chain,
            typically a free flyer (a.k.a. floating base) for mobile robots.

    Returns:
        Robot models for Pinocchio.
    """
    return pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=package_dirs,
        root_joint=root_joint_type(),
    )


def load_robot_description(
    description_name: str,
    root_joint_type: PinocchioJoint = pin.JointModelFreeFlyer,
) -> pin.RobotWrapper:
    """
    Load a robot description from `robot_descriptions
    <https://pypi.org/project/robot_descriptions/>`__.

    Args:
        description_name: Name of the robot description.
        root_joint_type: Type of the first joint of the kinematic chain,
            typically a free flyer (a.k.a. floating base) for mobile robots.

    Returns:
        Robot models for Pinocchio.
    """
    if robot_descriptions is None:
        raise ImportError(
            "robot_descriptions not found, "
            "try ``pip install robot_descriptions``"
        )
    module = import_module(f"robot_descriptions.{description_name}")
    robot = pin.RobotWrapper.BuildFromURDF(
        filename=module.URDF_PATH,
        package_dirs=[
            module.MESHES_PATH,
            module.PACKAGE_PATH,
            module.REPOSITORY_PATH,
            os.path.dirname(module.PACKAGE_PATH),
            os.path.dirname(module.REPOSITORY_PATH),
        ],
        root_joint=root_joint_type(),
    )
    return robot
