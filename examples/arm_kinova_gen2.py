#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Kinova Gen2 arm tracking a moving target."""

import numpy as np
import qpsolvers

import meshcat_shapes
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.utils import custom_configuration_vector
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
    robot = load_robot_description("gen2_description", root_joint=None)

    viz = start_meshcat_visualizer(robot)
    viewer = viz.viewer
    meshcat_shapes.frame(viewer["end_effector_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["end_effector"], opacity=1.0)

    end_effector_task = FrameTask(
        "j2s6s200_end_effector",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
    )

    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )

    tasks = [end_effector_task, posture_task]

    q_ref = custom_configuration_vector(
        robot,
        j2s6s200_joint_2=1.0,
        j2s6s200_joint_3=1.0,
        j2s6s200_joint_5=1.0,
    )
    configuration = pink.Configuration(robot.model, robot.data, q_ref)
    for task in tasks:
        task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update task targets
        end_effector_target = end_effector_task.transform_target_to_world
        end_effector_target.translation[1] = 0.8 + 0.1 * np.sin(2.0 * t)
        end_effector_target.translation[2] = 0.2

        # Update visualization frames
        viewer["end_effector_target"].set_transform(end_effector_target.np)
        viewer["end_effector"].set_transform(
            configuration.get_transform_frame_to_world(
                end_effector_task.frame
            ).np
        )

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
