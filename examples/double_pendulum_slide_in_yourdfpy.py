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
Raise the double pendulum up and down.
"""

from os import path
from functools import partial

import pinocchio as pin
import yourdfpy

import pink


def callback(scene, robot, viz):
    configuration = pink.apply_configuration(robot, robot.q0)
    viz.update_cfg(configuration.q)


if __name__ == "__main__":
    urdf_path = path.join(path.dirname(__file__), "double_pendulum.urdf")
    robot = pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=["."],
        root_joint=None,
    )

    viz = yourdfpy.URDF.load(urdf_path)
    viz.show(callback=partial(callback, robot=robot, viz=viz))
