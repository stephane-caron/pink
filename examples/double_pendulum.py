#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Swing the double pendulum left and right."""

import os

import numpy as np
import pink
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer

import meshcat_shapes

if __name__ == "__main__":
    # Load robot description
    urdf_path = os.path.join(
        os.path.dirname(__file__),
        "robots",
        "double_pendulum.urdf",
    )
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
        "tip": FrameTask(
            "link3",
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1e-3,  # [cost] / [rad]
        ),
        "posture": PostureTask(
            cost=1e-2,  # [cost] / [rad]
        ),
    }

    # Initialize tasks from the initial configuration
    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    for task in tasks.values():
        task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    # Homework: what happens if we replace -= by += in the following line?
    tasks["tip"].transform_target_to_world.translation[2] -= 0.1

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "quadprog" in qpsolvers.available_solvers:
        solver = "quadprog"

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
            configuration.get_transform_frame_to_world(tasks["tip"].body).np
        )

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks.values(), dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
