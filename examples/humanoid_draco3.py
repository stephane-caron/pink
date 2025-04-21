#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

"""DRACO 3 humanoid standing on two feet and reaching with a hand."""

import meshcat_shapes
import numpy as np
import pinocchio as pin
import qpsolvers

import pink
from pink import solve_ik
from pink.tasks import FrameTask, JointCouplingTask, PostureTask
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
    ) from exc


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
        T.translation[1] += 0.1 + 0.05 * np.sin(8.0 * t)
        T.translation[2] += 0.5
        return T


if __name__ == "__main__":
    robot = load_robot_description(
        "draco3_description", root_joint=pin.JointModelFreeFlyer()
    )

    # Initialize visualization
    viz = start_meshcat_visualizer(robot)
    wrist_frame = viz.viewer["right_wrist_pose"]
    meshcat_shapes.frame(wrist_frame)

    # Set initial robot configuration
    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    viz.display(configuration.q)

    # Tasks initialization for IK
    left_foot_task = FrameTask(
        "l_foot_contact",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    pelvis_task = FrameTask(
        "torso_com_link",
        position_cost=1.0,
        orientation_cost=0.0,
    )
    right_foot_task = FrameTask(
        "r_foot_contact",
        position_cost=1.0,
        orientation_cost=1.0,
    )
    right_wrist_task = FrameTask(
        "r_hand_contact",
        position_cost=4.0,
        orientation_cost=4.0,
    )
    posture_task = PostureTask(
        cost=1e-1,  # [cost] / [rad]
    )

    # Joint coupling task
    r_knee_holonomic_task = JointCouplingTask(
        ["r_knee_fe_jp", "r_knee_fe_jd"],
        [1.0, -1.0],
        100.0,
        configuration,
        lm_damping=1e-7,
    )
    l_knee_holonomic_task = JointCouplingTask(
        ["l_knee_fe_jp", "l_knee_fe_jd"],
        [1.0, -1.0],
        100.0,
        configuration,
        lm_damping=1e-7,
    )

    tasks = [
        left_foot_task,
        pelvis_task,
        right_foot_task,
        right_wrist_task,
        posture_task,
        l_knee_holonomic_task,
        r_knee_holonomic_task,
    ]

    # Task target specifications
    pelvis_pose = configuration.get_transform_frame_to_world(
        "torso_com_link"
    ).copy()
    pelvis_pose.translation[0] += 0.05
    pelvis_task.set_target(pelvis_pose)

    transform_l_ankle_target_to_init = pin.SE3(
        np.eye(3), np.array([0.1, 0.0, 0.0])
    )
    transform_r_ankle_target_to_init = pin.SE3(
        np.eye(3), np.array([-0.1, 0.0, 0.0])
    )

    left_foot_task.set_target(
        configuration.get_transform_frame_to_world("l_foot_contact")
        * transform_l_ankle_target_to_init
    )
    right_foot_task.set_target(
        configuration.get_transform_frame_to_world("r_foot_contact")
        * transform_r_ankle_target_to_init
    )

    pelvis_task.set_target(
        configuration.get_transform_frame_to_world("torso_com_link")
    )

    posture_task.set_target_from_configuration(configuration)

    right_wrist_pose = WavingPose(
        configuration.get_transform_frame_to_world("r_hand_contact")
    )

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

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
