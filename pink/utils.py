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

from typing import List

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


def load_urdf(
    urdf_path: str,
    package_dirs: List[str] = [],
    free_flyer: bool = True,
) -> pin.RobotWrapper:
    """
    Load the URDF of a robot description.

    Args:
        urdf_path: Path to the URDF file.
        package_dirs: List of directories Pinocchio can use as guesses to
            replace relative and ``package://`` paths from the URDF.
        free_flyer: Add a free flyer (a.k.a. floating base) as the first joint
            of the kinematic chain.
    """
    return pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=package_dirs,
        root_joint=pin.JointModelFreeFlyer() if free_flyer else None,
    )
