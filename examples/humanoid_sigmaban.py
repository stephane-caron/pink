#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

"""SigmaBan humanoid standing on two feet."""

import numpy as np
import pinocchio as pin
import qpsolvers

import meshcat_shapes
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
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
        "sigmaban_description", root_joint=pin.JointModelFreeFlyer()
    )

    # Initialize visualization
    viz = start_meshcat_visualizer(robot)
    viewer = viz.viewer
    meshcat_shapes.frame(viewer["left_foot_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["right_foot_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["torso_target"], opacity=0.5)

    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    viz.display(configuration.q)

    left_foot_task = FrameTask(
        "left_foot_tip",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    torso_task = FrameTask(
        "torso",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    right_foot_task = FrameTask(
        "right_foot_tip",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    posture_task = PostureTask(
        cost=1e-2,  # [cost] / [rad]
    )
    tasks = [left_foot_task, torso_task, right_foot_task, posture_task]

    torso_pose = configuration.get_transform_frame_to_world("torso").copy()
    # torso_pose.translation[0] += 0.05
    torso_task.set_target(torso_pose)
    posture_task.set_target_from_configuration(configuration)

    transform_left_foot_tip_target_to_init = pin.SE3(
        np.eye(3), np.array([0.0, 0.03, 0.0])
    )
    transform_right_foot_tip_target_to_init = pin.SE3(
        np.eye(3), np.array([0.0, -0.03, 0.0])
    )

    left_foot_task.set_target(
        configuration.get_transform_frame_to_world("left_foot_tip")
        * transform_left_foot_tip_target_to_init
    )
    right_foot_task.set_target(
        configuration.get_transform_frame_to_world("right_foot_tip")
        * transform_right_foot_tip_target_to_init
    )
    torso_task.set_target(configuration.get_transform_frame_to_world("torso"))

    # Display targets
    viewer["left_foot_target"].set_transform(
        left_foot_task.transform_target_to_world.np
    )
    viewer["right_foot_target"].set_transform(
        right_foot_task.transform_target_to_world.np
    )
    viewer["torso_target"].set_transform(
        torso_task.transform_target_to_world.np
    )

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
