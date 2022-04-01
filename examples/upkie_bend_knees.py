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
Upkie wheeled biped bending its knees.
"""

import os
import time

import numpy as np
import pinocchio as pin

import pink
import pink.models

from pink import solve_ik
from pink.tasks import BodyTask, PostureTask
from pink.utils import custom_configuration_vector

import meshcat_shapes


class ElevatorPose:

    """
    Frame that goes up and down with respect to the world.
    """

    def __init__(self, init: pin.SE3):
        """
        Initialize pose.

        Args:
            init: Initial transform from the wrist frame to the world frame.
        """
        self.init = init

    def at(self, t):
        T = self.init.copy()
        T.translation[2] += 0.1 * np.sin(t)
        return T


if __name__ == "__main__":
    models_dir = os.path.join(os.path.dirname(__file__), "../tests", "models")
    upkie_description = os.path.join(models_dir, "upkie_description")
    robot = pink.models.build_from_urdf(upkie_description)
    viz = pin.visualize.MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model
    )
    robot.setVisualizer(viz, init=False)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    configuration = pink.apply_configuration(robot, robot.q0)
    viz.display(configuration.q)

    tasks = {
        "base": BodyTask(
            "base",
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
        ),
        "left_contact": BodyTask(
            "left_contact",
            position_cost=[0.1, 0.0, 0.1],  # [cost] / [m]
            orientation_cost=0.0,  # [cost] / [rad]
        ),
        "right_contact": BodyTask(
            "right_contact",
            position_cost=[0.1, 0.0, 0.1],  # [cost] / [m]
            orientation_cost=0.0,  # [cost] / [rad]
        ),
        "posture": PostureTask(
            cost=1e-3,  # [cost] / [rad]
        ),
    }

    tasks["posture"].set_target(
        custom_configuration_vector(robot, left_knee=0.2, right_knee=-0.2)
    )

    for body, task in tasks.items():
        if type(task) is BodyTask:
            task.set_target(configuration.get_transform_body_to_world(body))

    left_contact_target = ElevatorPose(
        configuration.get_transform_body_to_world("left_contact")
    )
    right_contact_target = ElevatorPose(
        configuration.get_transform_body_to_world("right_contact")
    )

    left_contact_frame = viz.viewer["left_contact_frame"]
    meshcat_shapes.set_frame(left_contact_frame)
    right_contact_frame = viz.viewer["right_contact_frame"]
    meshcat_shapes.set_frame(right_contact_frame)

    dt = 5e-3  # [s]
    for t in np.arange(0.0, 10.0, dt):
        tasks["left_contact"].set_target(left_contact_target.at(t))
        tasks["right_contact"].set_target(right_contact_target.at(t))
        left_contact_frame.set_transform(left_contact_target.at(t).np)
        right_contact_frame.set_transform(right_contact_target.at(t).np)
        velocity = solve_ik(configuration, tasks.values(), dt)
        q = configuration.integrate(velocity, dt)
        configuration = pink.apply_configuration(robot, q)
        viz.display(q)
        time.sleep(dt)  # TODO(scaron): proper rate
