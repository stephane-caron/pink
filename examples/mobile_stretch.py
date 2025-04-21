#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2023 Inria

"""Move a Stretch RE1 with a fixed fingertip target around the origin."""

import numpy as np
import pinocchio as pin
import qpsolvers

import meshcat_shapes
import pink
from pink import PinkError, solve_ik
from pink.tasks import FrameTask
from pink.visualization import start_meshcat_visualizer

try:
    from loop_rate_limiters import RateLimiter
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples use loop rate limiters, "
        "try `[conda|pip] install loop-rate-limiters`"
    ) from exc

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try `[conda|pip] install robot_descriptions`"
    )

# Trajectory parameters to play with ;)
circle_radius = 0.5  # [m]
fingertip_height = 0.7  # [m]

if __name__ == "__main__":
    robot = load_robot_description(
        "stretch_description", root_joint=pin.JointModelPlanar()
    )

    # Initialize visualization
    viz = start_meshcat_visualizer(robot)
    viewer = viz.viewer
    meshcat_shapes.frame(viewer["base_target_frame"], opacity=0.5)
    meshcat_shapes.frame(viewer["fingertip_target_frame"], opacity=1.0)

    # Define tasks
    base_task = FrameTask(
        "base_link",
        position_cost=0.1,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
    )
    fingertip_task = FrameTask(
        "link_gripper_fingertip_right",
        position_cost=1.0,
        orientation_cost=1e-4,
    )
    tasks = [base_task, fingertip_task]

    # Initialize tasks from the initial configuration
    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    base_task.set_target_from_configuration(configuration)
    transform_fingertip_target_to_world = pin.SE3(
        rotation=np.eye(3), translation=np.array([0.0, 0.0, fingertip_height])
    ) * configuration.get_transform_frame_to_world(fingertip_task.frame)
    center_translation = transform_fingertip_target_to_world.translation[:2]
    fingertip_task.set_target(transform_fingertip_target_to_world)
    viz.display(configuration.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "quadprog" in qpsolvers.available_solvers:
        solver = "quadprog"

    rate = RateLimiter(frequency=100.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update task targets
        T = base_task.transform_target_to_world
        u = np.array([np.cos(t), np.sin(t)])
        T.translation[:2] = center_translation + circle_radius * u
        T.rotation = pin.utils.rpyToMatrix(0.0, 0.0, 0.5 * np.pi * t)

        # Update visualizer frames
        viewer["base_target_frame"].set_transform(T.np)
        viewer["fingertip_target_frame"].set_transform(
            fingertip_task.transform_target_to_world.np
        )
        viewer["base_frame"].set_transform(
            configuration.get_transform_frame_to_world(base_task.frame).np
        )
        viewer["fingertip_frame"].set_transform(
            configuration.get_transform_frame_to_world(fingertip_task.frame).np
        )

        # Compute velocity and integrate it into next configuration
        try:
            velocity = solve_ik(configuration, tasks, dt, solver=solver)
        except PinkError as exn:
            if solver != "quadprog":
                raise PinkError(
                    "IK failed as detailed in the traceback above. "
                    f"Note that `solve_ik` was called with {solver=}, "
                    "but this example works better with solver='quadprog'."
                ) from exn
            raise exn
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
