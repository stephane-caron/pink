#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2023 Inria

"""Move a Stretch RE1 with a fingertip target in the mobile-base frame."""

import meshcat_shapes
import numpy as np
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter

import pink
from pink import solve_ik
from pink.tasks import FrameTask, RelativeFrameTask
from pink.visualization import start_meshcat_visualizer

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try ``pip install robot_descriptions``"
    )

# Trajectory parameters to play with ;)
CIRCLE_RADIUS = 0.5  # [m]
FINGERTIP_HEIGHT = 0.7  # [m]

if __name__ == "__main__":
    robot = load_robot_description(
        "stretch_description", root_joint=pin.JointModelPlanar()
    )
    # __import__("IPython").embed()

    # Initialize visualizer
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
    fingertip_task = RelativeFrameTask(
        "link_gripper_fingertip_right",
        "base_link",
        position_cost=1.0,
        orientation_cost=1e-4,
    )
    tasks = [base_task, fingertip_task]

    # Initialize tasks from the initial configuration
    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    base_task.set_target_from_configuration(configuration)
    transform_fingertip_target_to_base = pin.SE3(
        rotation=np.eye(3),
        translation=np.array([0.1, 0.0, FINGERTIP_HEIGHT]),
    )
    transform_fingertip_to_world = configuration.get_transform_frame_to_world(
        fingertip_task.frame
    )
    center_translation = transform_fingertip_to_world.translation[:2]
    fingertip_task.set_target(transform_fingertip_target_to_base)
    viz.display(configuration.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "quadprog" in qpsolvers.available_solvers:
        solver = "quadprog"

    rate = RateLimiter(frequency=100.0)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update base task target
        T = base_task.transform_target_to_world
        u = np.array([np.cos(t), np.sin(t)])
        T.translation[:2] = center_translation + CIRCLE_RADIUS * u
        T.rotation = pin.utils.rpyToMatrix(0.0, 0.0, 0.5 * np.pi * t)

        # Update fingertip target
        fingertip_task.transform_target_to_root.translation[2] = (
            FINGERTIP_HEIGHT + 0.1 * u[1]
        )

        # Update visualizer frames
        viewer["base_target_frame"].set_transform(T.np)
        transform_fingertip_target_to_world = (
            configuration.get_transform_frame_to_world("base_link")
            * fingertip_task.transform_target_to_root
        )
        viewer["fingertip_target_frame"].set_transform(
            transform_fingertip_target_to_world.np
        )
        viewer["base_frame"].set_transform(
            configuration.get_transform_frame_to_world(base_task.frame).np
        )
        viewer["fingertip_frame"].set_transform(
            configuration.get_transform_frame_to_world(fingertip_task.frame).np
        )

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        # input("next?")
        t += dt
