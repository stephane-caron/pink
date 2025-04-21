#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""JVRC-1 humanoid standing on two feet and reaching with a hand."""

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

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try `[conda|pip] install robot_descriptions`"
    )


class WavingPose:
    """Moving target to the wave the right hand."""

    def __init__(self, init: pin.SE3):
        """Initialize pose.

        Args:
            init: Initial transform from the wrist frame to the world frame.
        """
        self.init = init

    def at(self, t):
        """Get waving pose at a given time.

        Args:
            t: Time in seconds.
        """
        T = self.init.copy()
        R = T.rotation
        R = np.dot(R, pin.utils.rpyToMatrix(0.0, 0.0, np.pi / 2))
        R = np.dot(R, pin.utils.rpyToMatrix(0.0, -np.pi, 0.0))
        T.rotation = R
        T.translation[0] += 0.5
        T.translation[1] += -0.1 + 0.05 * np.sin(8.0 * t)
        T.translation[2] += 0.5
        return T


if __name__ == "__main__":
    robot = load_robot_description(
        "jvrc_description", root_joint=pin.JointModelFreeFlyer()
    )

    # Initialize visualization
    viz = start_meshcat_visualizer(robot)
    viewer = viz.viewer
    wrist_frame = viewer["right_wrist_pose"]
    meshcat_shapes.frame(viewer["pelvis_pose"])
    meshcat_shapes.frame(wrist_frame)

    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    viz.display(configuration.q)

    left_foot_task = FrameTask(
        "l_ankle", position_cost=1.0, orientation_cost=3.0
    )
    pelvis_task = FrameTask(
        "PELVIS_S", position_cost=1.0, orientation_cost=0.0
    )
    right_foot_task = FrameTask(
        "r_ankle", position_cost=1.0, orientation_cost=3.0
    )
    right_wrist_task = FrameTask(
        "r_wrist", position_cost=1.0, orientation_cost=3.0
    )
    tasks = [left_foot_task, pelvis_task, right_foot_task, right_wrist_task]

    pelvis_pose = configuration.get_transform_frame_to_world("PELVIS_S").copy()
    pelvis_pose.translation[0] += 0.05
    viewer["pelvis_pose"].set_transform(pelvis_pose.np)
    pelvis_task.set_target(pelvis_pose)

    transform_l_ankle_target_to_init = pin.SE3(
        np.eye(3), np.array([0.1, 0.0, 0.0])
    )
    transform_r_ankle_target_to_init = pin.SE3(
        np.eye(3), np.array([-0.1, 0.0, 0.0])
    )

    left_foot_task.set_target(
        configuration.get_transform_frame_to_world("l_ankle")
        * transform_l_ankle_target_to_init
    )
    right_foot_task.set_target(
        configuration.get_transform_frame_to_world("r_ankle")
        * transform_r_ankle_target_to_init
    )
    pelvis_task.set_target(
        configuration.get_transform_frame_to_world("PELVIS_S")
    )

    right_wrist_pose = WavingPose(
        configuration.get_transform_frame_to_world("r_wrist")
    )

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "proxqp" in qpsolvers.available_solvers:
        solver = "proxqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update task targets
        right_wrist_task.set_target(right_wrist_pose.at(t))
        wrist_frame.set_transform(right_wrist_pose.at(t).np)

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
