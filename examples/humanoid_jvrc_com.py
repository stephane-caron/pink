#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# dependencies = ["daqp", "loop-rate-limiters", "meshcat", "pin-pink",
# "qpsolvers", "robot_descriptions", "typing_extensions"]
# ///

"""JVRC-1 humanoid backbending by regulating its CoM position."""

import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter
from robot_descriptions.loaders.pinocchio import load_robot_description

import pink
from pink import solve_ik
from pink.tasks import ComTask, FrameTask
from pink.visualization import start_meshcat_visualizer

if __name__ == "__main__":
    robot = load_robot_description(
        "jvrc_description", root_joint=pin.JointModelFreeFlyer()
    )

    # Initialize visualization
    viz = start_meshcat_visualizer(robot)
    viz.viewer["com_target"].set_object(
        g.Sphere(0.03), g.MeshLambertMaterial(color=0x00FF00)
    )
    viz.viewer["com"].set_object(
        g.Sphere(0.03), g.MeshLambertMaterial(color=0xFF0000)
    )

    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    viz.display(configuration.q)

    left_ankle_task = FrameTask(
        "l_ankle", position_cost=1.0, orientation_cost=3.0
    )
    right_ankle_task = FrameTask(
        "r_ankle", position_cost=1.0, orientation_cost=3.0
    )
    com_task = ComTask(cost=2.0)

    transform_l_ankle_target_to_init = pin.SE3(
        np.eye(3), np.array([0.1, 0.0, 0.0])
    )
    transform_r_ankle_target_to_init = pin.SE3(
        np.eye(3), np.array([-0.1, 0.0, 0.0])
    )
    left_ankle_task.set_target(
        configuration.get_transform_frame_to_world("l_ankle")
        * transform_l_ankle_target_to_init
    )
    right_ankle_task.set_target(
        configuration.get_transform_frame_to_world("r_ankle")
        * transform_r_ankle_target_to_init
    )

    initial_com = pin.centerOfMass(robot.model, robot.data, configuration.q)
    com_task.set_target(initial_com)

    tasks = [com_task, left_ankle_task, right_ankle_task]

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"
    elif "osqp" in qpsolvers.available_solvers:
        solver = "osqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    period = 2
    omega = 2 * np.pi / period

    while True:
        pin.centerOfMass(robot.model, robot.data, configuration.q)
        com = robot.data.com[0]

        # Update CoM target
        Az = 0.05
        desired_com = initial_com.copy()
        desired_com[2] += Az * np.sin(omega * t)
        com_task.set_target(desired_com)

        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            safety_break=False,
        )
        configuration.integrate_inplace(velocity, dt)
        viz.display(configuration.q)
        viz.viewer["com_target"].set_transform(
            tf.translation_matrix(desired_com)
        )
        viz.viewer["com"].set_transform(tf.translation_matrix(com))

        rate.sleep()
        t += dt
