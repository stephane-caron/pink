#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Move an omnidirectional robot between two jumping targets.

Illustrates the notion of screw path.
"""

import os

import numpy as np
import pinocchio as pin
import qpsolvers

import meshcat_shapes
import pink
from pink import solve_ik
from pink.tasks import FrameTask
from pink.visualization import start_meshcat_visualizer

try:
    from loop_rate_limiters import RateLimiter
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples use loop rate limiters, "
        "try `[conda|pip] install loop-rate-limiters`"
    ) from exc

if __name__ == "__main__":
    # Load robot description
    urdf_path = os.path.join(
        os.path.dirname(__file__),
        "robots",
        "omnidirectional_wheeled_robot.urdf",
    )
    robot = pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=["."],
        root_joint=pin.JointModelPlanar(),
    )

    # Initialize visualization
    viz = start_meshcat_visualizer(robot)
    viewer = viz.viewer
    meshcat_shapes.frame(viewer["target_frame"], opacity=0.5)
    meshcat_shapes.frame(viewer["tip_frame"], opacity=1.0)

    # Define tasks
    base_task = FrameTask(
        "base",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1e-5,  # [cost] / [rad]
    )
    base_task.gain = 0.05  # slow things down
    tasks = [base_task]

    # Initialize tasks from the initial configuration
    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    base_task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=100.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    jump_period = 4.0
    while True:
        # Update task targets
        jumpy = 0.0 if (t / jump_period) % 1.0 <= 0.5 else -1.0
        T = base_task.transform_target_to_world
        T.translation[0] = 0.2 * jumpy
        T.rotation = pin.utils.rpyToMatrix(0.0, 0.0, np.pi * jumpy)

        # Update visualizer frames
        viewer["target_frame"].set_transform(T.np)
        viewer["tip_frame"].set_transform(
            configuration.get_transform_frame_to_world(base_task.frame).np
        )

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
