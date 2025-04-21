#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Ivan Domrachev, Simeon Nedelchev

"""G1 humanoid squat by regulating CoM"""

import numpy as np
import pinocchio as pin
import qpsolvers

import pink
from pink import solve_ik
from pink.tasks import ComTask, FrameTask, PostureTask
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
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try `[conda|pip] install robot_descriptions`"
    ) from exc  # noqa: E501


if __name__ == "__main__":
    robot = load_robot_description(
        "g1_description", root_joint=pin.JointModelFreeFlyer()
    )

    # Initialize visualization
    viz = start_meshcat_visualizer(robot)

    q_ref = np.zeros(robot.nq)
    q_ref[2] = 0.72
    q_ref[6] = 1.0

    configuration = pink.Configuration(robot.model, robot.data, q_ref)
    pelvis_orientation_task = FrameTask(
        "pelvis",
        position_cost=0.0,  # [cost] / [m]
        orientation_cost=10.0,  # [cost] / [rad]
    )

    com_task = ComTask(cost=200.0)
    com_task.set_target_from_configuration(configuration)

    posture_task = PostureTask(
        cost=1e-1,  # [cost] / [rad]
    )

    tasks = [pelvis_orientation_task, posture_task, com_task]

    for foot in ["right_ankle_roll_link", "left_ankle_roll_link"]:
        task = FrameTask(
            foot,
            position_cost=200.0,  # [cost] / [m]
            orientation_cost=10.0,  # [cost] / [rad]
        )
        tasks.append(task)

    for arm_points in ["right_palm_link", "left_palm_link"]:
        task = FrameTask(
            arm_points,
            position_cost=4.0,  # [cost] / [m]
            orientation_cost=0.0,  # [cost] / [rad]
        )
        tasks.append(task)

    for task in tasks:
        task.set_target_from_configuration(configuration)
        if isinstance(task, FrameTask):
            target = task.transform_target_to_world
            if task.frame in ["right_palm_link", "left_palm_link"]:
                target.translation += np.array([-0.1, 0.0, -0.2])
                task.set_target(target)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "osqp" in qpsolvers.available_solvers:
        solver = "osqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    period = 2
    omega = 2 * np.pi / period

    while True:
        # Update CoM target
        Az = 0.05
        desired_com = np.zeros(3)
        desired_com[2] = 0.55 + Az * np.sin(omega * t)
        com_task.set_target(desired_com)

        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            damping=0.01,
            safety_break=False,
        )
        configuration.integrate_inplace(velocity, dt)

        viz.display(configuration.q)
        rate.sleep()
        t += dt
