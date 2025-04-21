#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

import meshcat_shapes
import numpy as np
import pinocchio as pin
import qpsolvers

import pink
from pink import solve_ik
from pink.tasks import FrameTask, RollingTask
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


if __name__ == "__main__":
    robot = load_robot_description(
        "upkie_description",
        root_joint=pin.JointModelFreeFlyer(),
    )
    visualizer = start_meshcat_visualizer(robot)

    base_task = FrameTask(
        "base",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
    )
    left_wheel_rolling = RollingTask(
        "left_wheel",
        floor_frame="universe",
        wheel_radius=0.06,
        cost=10.0,
    )
    right_wheel_rolling = RollingTask(
        "right_wheel",
        floor_frame="universe",
        wheel_radius=0.06,
        cost=10.0,
    )
    left_wheel_position = FrameTask(
        "left_wheel",
        position_cost=1.0,
        orientation_cost=0.0,
    )
    right_wheel_position = FrameTask(
        "right_wheel",
        position_cost=1.0,
        orientation_cost=0.0,
    )

    init_x = -2.0  # in [m]
    q_init = robot.q0.copy()
    q_init[0] = init_x  # world x-axis
    q_init[2] = 0.56  # world z-axis
    configuration = pink.Configuration(robot.model, robot.data, q_init)
    base_task.set_target_from_configuration(configuration)
    left_wheel_position.set_target_from_configuration(configuration)
    right_wheel_position.set_target_from_configuration(configuration)
    visualizer.display(configuration.q)

    base_target = base_task.transform_target_to_world
    left_wheel_target = left_wheel_position.transform_target_to_world
    right_wheel_target = right_wheel_position.transform_target_to_world

    viewer = visualizer.viewer
    meshcat_shapes.frame(viewer["base"], opacity=1.0)
    meshcat_shapes.frame(viewer["base_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["left_wheel_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["right_wheel_target"], opacity=0.5)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "proxqp" in qpsolvers.available_solvers:
        solver = "proxqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]

    while True:
        # Update base task target
        base_x = 0.1 * t + init_x
        base_target.translation[0] = base_x
        left_wheel_target.translation[0] = base_x + 0.1 * np.sin(t)
        right_wheel_target.translation[0] = base_x - 0.1 * np.sin(t)

        # Update visualization frames
        viewer["base_target"].set_transform(base_target.np)
        viewer["left_wheel_target"].set_transform(left_wheel_target.np)
        viewer["right_wheel_target"].set_transform(right_wheel_target.np)
        viewer["base"].set_transform(
            configuration.get_transform_frame_to_world("base").np
        )

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(
            configuration,
            tasks=[base_task, left_wheel_position, right_wheel_position],
            dt=dt,
            solver=solver,
            damping=1e-3,
            constraints=[left_wheel_rolling, right_wheel_rolling],
        )
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        visualizer.display(configuration.q)
        rate.sleep()
        t += dt
