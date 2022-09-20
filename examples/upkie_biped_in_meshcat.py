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

import time

import numpy as np
import pinocchio as pin
from utils import add_meshcat_frame_axes

import pink
from pink import solve_ik
from pink.tasks import BodyTask, PostureTask
from pink.utils import custom_configuration_vector

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try `pip install robot_descriptions`"
    )


if __name__ == "__main__":
    robot = load_robot_description(
        "upkie_description", root_joint=pin.JointModelFreeFlyer()
    )
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
            task.set_target_from_configuration(configuration)

    left_contact_target = configuration.get_transform_body_to_world(
        "left_contact"
    )
    right_contact_target = configuration.get_transform_body_to_world(
        "right_contact"
    )

    left_contact_frame = viz.viewer["left_contact_frame"]
    add_meshcat_frame_axes(left_contact_frame)
    right_contact_frame = viz.viewer["right_contact_frame"]
    add_meshcat_frame_axes(right_contact_frame)

    dt = 5e-3  # [s]
    for t in np.arange(0.0, 10.0, dt):
        left_contact_target.translation[2] += 0.1 * np.sin(t) * dt
        right_contact_target.translation[2] += 0.1 * np.sin(t) * dt
        tasks["left_contact"].set_target(left_contact_target)
        tasks["right_contact"].set_target(right_contact_target)
        left_contact_frame.set_transform(left_contact_target.np)
        right_contact_frame.set_transform(right_contact_target.np)
        velocity = solve_ik(configuration, tasks.values(), dt)
        q = configuration.integrate(velocity, dt)
        configuration = pink.apply_configuration(robot, q)
        viz.display(q)
        time.sleep(dt)  # TODO(scaron): proper rate
