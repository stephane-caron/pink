#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Stéphane Caron.
#
# This file is part of Pink.
#
# Pink is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Pink is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Pink. If not, see <http://www.gnu.org/licenses/>.

"""
Humanoid robot model standing on two feet and reaching with a hand.
"""

import time

import numpy as np
import pinocchio as pin

import pink
import pink.models

from pink.tasks import BodyTask
from pink import solve_ik

from meshcat_frame import set_frame


class TransformToWorld(pin.SE3):

    pass


class WavingPose:
    def __init__(self, init: TransformToWorld):
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
        R = np.dot(R, pin.utils.rpyToMatrix(0.0, np.pi, 0.0))
        T.rotation = R
        T.translation[1] += -0.1 + 0.05 * np.sin(8.0 * t)
        T.translation[2] += 0.5
        return T


if __name__ == "__main__":
    robot = pink.models.build_jvrc_model()
    viz = pin.visualize.MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model
    )
    robot.setVisualizer(viz, init=False)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    configuration = pink.apply_configuration(robot, robot.q0)
    viz.display(configuration.q)
    dt = 5e-3

    left_foot_task = BodyTask(
        "l_ankle", position_cost=1.0, orientation_cost=3.0
    )
    pelvis_task = BodyTask(
        "PELVIS_S", position_cost=1.0, orientation_cost=0.0
    )
    right_foot_task = BodyTask(
        "r_ankle", position_cost=1.0, orientation_cost=3.0
    )
    right_wrist_task = BodyTask(
        "r_wrist", position_cost=1.0, orientation_cost=3.0
    )
    tasks = [left_foot_task, pelvis_task, right_foot_task, right_wrist_task]

    pelvis_pose = configuration.get_transform_body_to_world("PELVIS_S").copy()
    pelvis_pose.translation[0] += 0.05
    set_frame(viz.viewer["pelvis_pose"])
    viz.viewer["pelvis_pose"].set_transform(pelvis_pose.np)
    pelvis_task.set_target(pelvis_pose)

    left_foot_task.set_target(
        configuration.get_transform_body_to_world("l_ankle")
    )
    right_foot_task.set_target(
        configuration.get_transform_body_to_world("r_ankle")
    )

    right_wrist_pose = WavingPose(
        configuration.get_transform_body_to_world("r_wrist")
    )

    wrist_frame = viz.viewer["right_wrist_pose"]
    set_frame(wrist_frame)

    dt = 5e-3  # [s]
    for t in np.arange(0.0, 10.0, dt):
        right_wrist_task.set_target(right_wrist_pose.at(t))
        wrist_frame.set_transform(right_wrist_pose.at(t).np)
        velocity = solve_ik(configuration, tasks, dt)
        q = configuration.integrate(velocity, dt)
        configuration = pink.apply_configuration(robot, q)
        viz.display(q)
        time.sleep(dt)  # TODO(scaron): proper rate
