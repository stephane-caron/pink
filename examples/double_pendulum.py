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
Swing the double pendulum left and right.
"""

from os import path

import meshcat_shapes
import numpy as np
import pinocchio as pin

import pink
from pink import solve_ik
from pink.tasks import BodyTask, PostureTask
from pink.utils import RateLimiter
from pink.visualization import start_meshcat_visualizer

if __name__ == "__main__":

    # Load robot description
    urdf_path = path.join(path.dirname(__file__), "double_pendulum.urdf")
    robot = pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=["."],
        root_joint=None,
    )

    # Initialize visualizer
    viz = start_meshcat_visualizer(robot)
    viewer = viz.viewer
    meshcat_shapes.frame(viewer["target_frame"], opacity=0.5)
    meshcat_shapes.frame(viewer["tip_frame"], opacity=1.0)

    # Define tasks
    tasks = {
        "tip": BodyTask(
            "link3",
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1e-3,  # [cost] / [rad]
        ),
        "posture": PostureTask(
            cost=1e-2,  # [cost] / [rad]
        ),
    }

    # Initialize tasks from the initial configuration
    configuration = pink.apply_configuration(robot, robot.q0)
    for task in tasks.values():
        task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    # Homework: what happens if we replace -= by += in the following line?
    tasks["tip"].transform_target_to_world.translation[2] -= 0.1

    rate = RateLimiter(frequency=100.0)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update task targets
        T = tasks["tip"].transform_target_to_world
        T.translation[1] = 0.1 * np.sin(t)

        # Update visualizer frames
        viewer["target_frame"].set_transform(T.np)
        viewer["tip_frame"].set_transform(
            configuration.get_transform_body_to_world(tasks["tip"].body).np
        )

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks.values(), dt)
        q = configuration.integrate(velocity, dt)
        configuration = pink.apply_configuration(robot, q)

        # Visualize result at fixed FPS
        viz.display(q)
        rate.sleep()
        t += dt
