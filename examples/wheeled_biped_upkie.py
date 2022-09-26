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

import meshcat_shapes
import numpy as np
import pinocchio as pin

import pink
from pink import solve_ik
from pink.tasks import BodyTask, PostureTask
from pink.utils import RateLimiter, custom_configuration_vector
from pink.visualization import start_meshcat_visualizer

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try ``pip install robot_descriptions``"
    )


if __name__ == "__main__":
    robot = load_robot_description(
        "upkie_description", root_joint=pin.JointModelFreeFlyer()
    )
    viz = start_meshcat_visualizer(robot)

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

    configuration = pink.apply_configuration(robot, robot.q0)
    for body, task in tasks.items():
        if type(task) is BodyTask:
            task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    left_contact_target = tasks["left_contact"].transform_target_to_world
    right_contact_target = tasks["right_contact"].transform_target_to_world

    viewer = viz.viewer
    meshcat_shapes.draw_frame(viewer["left_contact_target"], opacity=0.5)
    meshcat_shapes.draw_frame(viewer["right_contact_target"], opacity=0.5)
    meshcat_shapes.draw_frame(viewer["left_contact"], opacity=1.0)
    meshcat_shapes.draw_frame(viewer["right_contact"], opacity=1.0)

    rate = RateLimiter(frequency=200.0)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update task targets
        left_contact_target.translation[2] += 0.1 * np.sin(t) * dt
        right_contact_target.translation[2] += 0.1 * np.sin(t) * dt

        # Update visualization frames
        viewer["left_contact_target"].set_transform(left_contact_target.np)
        viewer["right_contact_target"].set_transform(right_contact_target.np)
        for body in ["left_contact", "right_contact"]:
            viewer[body].set_transform(
                configuration.get_transform_body_to_world(body).np
            )

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks.values(), dt)
        q = configuration.integrate(velocity, dt)
        configuration = pink.apply_configuration(robot, q)

        # Visualize result at fixed FPS
        viz.display(q)
        rate.sleep()
        t += dt
