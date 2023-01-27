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

"""Move an omnidirectional robot between two jumping targets.

Illustrates the notion of screw path.
"""

import os

import meshcat_shapes
import numpy as np
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter

import pink
from pink import solve_ik
from pink.tasks import BodyTask, PostureTask
from pink.visualization import start_meshcat_visualizer

if __name__ == "__main__":

    # Load robot description
    urdf_path = os.path.join(
        os.path.dirname(__file__),
        "robots",
        "omnidirectional_three_wheeled_robot.urdf",
    )
    robot = pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=["."],
        root_joint=pin.JointModelPlanar(),
    )

    # Initialize visualizer
    viz = start_meshcat_visualizer(robot)
    viewer = viz.viewer
    meshcat_shapes.frame(viewer["target_frame"], opacity=0.5)
    meshcat_shapes.frame(viewer["tip_frame"], opacity=1.0)

    # Define tasks
    base_task = BodyTask(
        "base",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1e-5,  # [cost] / [rad]
    )
    base_task.gain = 0.1
    posture_task = PostureTask(
        cost=1e-2,  # [cost] / [rad]
    )
    tasks = [base_task, posture_task]

    # Initialize tasks from the initial configuration
    configuration = pink.apply_configuration(robot, robot.q0)
    base_task.set_target_from_configuration(configuration)
    posture_task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "quadprog" in qpsolvers.available_solvers:
        solver = "quadprog"

    rate = RateLimiter(frequency=100.0)
    dt = rate.period
    t = 0.0  # [s]
    jump_period = 5.0
    while True:
        # Update task targets
        jumpy = 0.0 if (t / jump_period) % 1.0 <= 0.5 else -1.0
        T = base_task.transform_target_to_world
        T.translation[0] = 0.2 * jumpy
        T.rotation = pin.utils.rpyToMatrix(0.0, 0.0, np.pi * jumpy)

        # Update visualizer frames
        viewer["target_frame"].set_transform(T.np)
        viewer["tip_frame"].set_transform(
            configuration.get_transform_body_to_world(base_task.body).np
        )

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        q = configuration.integrate(velocity, dt)
        configuration = pink.apply_configuration(robot, q)

        # Visualize result at fixed FPS
        viz.display(q)
        rate.sleep()
        t += dt
