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
The JVRC model is a virtual humanoid robot that was released for education and
research during the Japanese Virtual Robotics Challenge in 2015.
"""

import os

import pinocchio as pin


def build_jvrc_from_urdf(path: str) -> pin.RobotWrapper:
    """
    Build the JVRC model from its URDF.

    Args:
        path: Path to a `jvrc_description`_ folder.

    .. _jvrc_description: https://github.com/stephane-caron/jvrc_description
    """
    abspath = os.path.abspath(path)
    return pin.RobotWrapper.BuildFromURDF(
        filename=os.path.join(abspath, "urdf", "jvrc1.urdf"),
        package_dirs=[os.path.dirname(abspath)],
        root_joint=pin.JointModelFreeFlyer(),
    )
