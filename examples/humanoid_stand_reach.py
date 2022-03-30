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
Humanoid robot model standing on two feet and reaching with a hand.
"""

import os
import time

import numpy as np
import pinocchio as pin

import pink
import pink.models

from pink.tasks import BodyTask
from pink import solve_ik

import meshcat_shapes


class WavingPose:
    def __init__(self, init: pin.SE3):
        """
        Initialize pose.

        Args:
            init: Initial transform from the wrist frame to the world frame.
        """
        self.init = init

    def at(self, t):
        T = self.init.copy()
        R = T.rotation
        R = np.dot(R, pin.utils.rpyToMatrix(0.0, 0.0, np.pi / 2))
        R = np.dot(R, pin.utils.rpyToMatrix(0.0, -np.pi, 0.0))
        T.rotation = R
        T.translation[0] += 0.5
        T.translation[1] += -0.1 + 0.05 * np.sin(8.0 * t)
        T.translation[2] += 0.5
        return T


if __name__ == "__main__":
    models_dir = os.path.join(os.path.dirname(__file__), "../tests", "models")
    jvrc_description = os.path.join(models_dir, "jvrc_description")
    robot = pink.models.build_from_urdf(jvrc_description)
    viz = pin.visualize.MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model
    )
    robot.setVisualizer(viz, init=False)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    configuration = pink.apply_configuration(robot, robot.q0)
    viz.display(configuration.q)

    left_foot_task = BodyTask(
        "l_ankle", position_cost=1.0, orientation_cost=3.0
    )
    pelvis_task = BodyTask("PELVIS_S", position_cost=1.0, orientation_cost=0.0)
    right_foot_task = BodyTask(
        "r_ankle", position_cost=1.0, orientation_cost=3.0
    )
    right_wrist_task = BodyTask(
        "r_wrist", position_cost=1.0, orientation_cost=3.0
    )
    tasks = [left_foot_task, pelvis_task, right_foot_task, right_wrist_task]

    pelvis_pose = configuration.get_transform_body_to_world("PELVIS_S").copy()
    pelvis_pose.translation[0] += 0.05
    meshcat_shapes.set_frame(viz.viewer["pelvis_pose"])
    viz.viewer["pelvis_pose"].set_transform(pelvis_pose.np)
    pelvis_task.set_target(pelvis_pose)

    transform_l_ankle_target_to_init = pin.SE3(
        np.eye(3), np.array([0.1, 0.0, 0.0])
    )
    transform_r_ankle_target_to_init = pin.SE3(
        np.eye(3), np.array([-0.1, 0.0, 0.0])
    )

    left_foot_task.set_target(
        configuration.get_transform_body_to_world("l_ankle")
        * transform_l_ankle_target_to_init
    )
    right_foot_task.set_target(
        configuration.get_transform_body_to_world("r_ankle")
        * transform_r_ankle_target_to_init
    )
    pelvis_task.set_target(
        configuration.get_transform_body_to_world("PELVIS_S")
    )

    right_wrist_pose = WavingPose(
        configuration.get_transform_body_to_world("r_wrist")
    )

    wrist_frame = viz.viewer["right_wrist_pose"]
    meshcat_shapes.set_frame(wrist_frame)

    dt = 5e-3  # [s]
    for t in np.arange(0.0, 10.0, dt):
        right_wrist_task.set_target(right_wrist_pose.at(t))
        wrist_frame.set_transform(right_wrist_pose.at(t).np)
        velocity = solve_ik(configuration, tasks, dt)
        q = configuration.integrate(velocity, dt)
        configuration = pink.apply_configuration(robot, q)
        viz.display(q)
        time.sleep(dt)  # TODO(scaron): proper rate
