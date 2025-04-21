#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Yumi two-armed manipulator with definition of custom frame and
collision avoidance w.r.t. those frames."""

import numpy as np
import pinocchio as pin
import qpsolvers

import meshcat_shapes
import pink
from pink import solve_ik
from pink.barriers import BodySphericalBarrier
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
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try `[conda|pip] install robot_descriptions`"
    ) from exc  # noqa: E501


if __name__ == "__main__":
    # Load the robot and define the custom frames
    robot = load_robot_description("yumi_description", root_joint=None)

    # Left arm frame
    l_frame_placement = pin.SE3()
    l_frame_placement.translation = np.array([0.0, 0.0, 0.05])
    l_frame_placement.rotation = np.eye(3)

    l_frame = pin.Frame(
        "yumi_barrier_l",
        robot.model.getJointId("yumi_joint_6_l"),
        robot.model.getFrameId("yumi_link_7_l"),
        l_frame_placement,
        pin.FrameType.OP_FRAME,
    )

    robot.model.addFrame(l_frame)

    # Right arm frame
    r_frame_placement = pin.SE3()
    r_frame_placement.translation = np.array([0.0, 0.0, 0.05])
    r_frame_placement.rotation = np.eye(3)

    r_frame = pin.Frame(
        "yumi_barrier_r",
        robot.model.getJointId("yumi_joint_6_r"),
        robot.model.getFrameId("yumi_link_7_r"),
        r_frame_placement,
        pin.FrameType.OP_FRAME,
    )

    robot.model.addFrame(r_frame)
    # Redefining robot data to take into account the new frames
    robot.data = pin.Data(robot.model)

    viz = start_meshcat_visualizer(robot)

    # Pink tasks
    left_end_effector_task = FrameTask(
        "yumi_link_7_l",
        position_cost=50.0,  # [cost] / [m]
        orientation_cost=10.0,  # [cost] / [rad]
    )
    right_end_effector_task = FrameTask(
        "yumi_link_7_r",
        position_cost=50.0,  # [cost] / [m]
        orientation_cost=10.0,  # [cost] / [rad]
    )

    # Pink barriers
    ee_barrier = BodySphericalBarrier(
        ("yumi_barrier_l", "yumi_barrier_r"),
        d_min=0.25,
        gain=100.0,
        safe_displacement_gain=1.0,
    )

    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )

    cbf_list = [ee_barrier]
    tasks = [left_end_effector_task, right_end_effector_task, posture_task]

    q_ref = np.array(
        [
            0.045,
            -0.155,
            -0.394,
            -0.617,
            -0.939,
            -0.343,
            -1.216,
            0,
            0,
            -0.374,
            -0.249,
            0.562,
            -0.520,
            0.934,
            -0.337,
            1.400,
            0,
            0,
        ]
    )
    configuration = pink.Configuration(robot.model, robot.data, q_ref)
    for task in tasks:
        task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    viewer = viz.viewer
    meshcat_shapes.frame(viewer["left_end_effector"], opacity=1.0)
    meshcat_shapes.sphere(
        viewer["left_ee_barrier"],
        opacity=0.4,
        color=0xFF0000,
        radius=0.125,
    )
    meshcat_shapes.sphere(
        viewer["right_ee_barrier"],
        opacity=0.4,
        color=0x00FF00,
        radius=0.125,
    )
    meshcat_shapes.frame(viewer["right_end_effector"], opacity=1.0)
    meshcat_shapes.frame(viewer["left_end_effector_target"], opacity=1.0)
    meshcat_shapes.frame(viewer["right_end_effector_target"], opacity=1.0)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "osqp" in qpsolvers.available_solvers:
        solver = "osqp"

    rate = RateLimiter(frequency=200.0)
    dt = rate.period
    t = 0.0  # [s]
    l_y_des = np.array([0.392, 0.392, 0.6])
    r_y_des = np.array([0.392, 0.392, 0.6])
    l_dy_des = np.zeros(3)
    r_dy_des = np.zeros(3)

    while True:
        # Calculate desired trajectory
        A = 0.1
        B = 0.1
        # z -- 0.4 - 0.8
        l_y_des[:] = (
            0.6,
            0.1 + B * np.sin(t),
            0.6 + A * np.sin(t),
        )
        r_y_des[:] = (
            0.6,
            -0.1 - B * np.sin(t),
            0.6 + A * np.sin(t),
        )
        l_dy_des[:] = 0, B * np.cos(t), A * np.cos(t)
        r_dy_des[:] = 0, -B * np.cos(t), A * np.cos(t)

        left_end_effector_task.transform_target_to_world.translation = l_y_des
        right_end_effector_task.transform_target_to_world.translation = r_y_des

        # Update visualization frames
        viewer["left_end_effector"].set_transform(
            configuration.get_transform_frame_to_world(
                left_end_effector_task.frame
            ).np
        )
        viewer["right_end_effector"].set_transform(
            configuration.get_transform_frame_to_world(
                right_end_effector_task.frame
            ).np
        )
        viewer["left_end_effector_target"].set_transform(
            left_end_effector_task.transform_target_to_world.np
        )
        viewer["right_end_effector_target"].set_transform(
            right_end_effector_task.transform_target_to_world.np
        )

        lb = configuration.get_transform_frame_to_world("yumi_barrier_l")
        rb = configuration.get_transform_frame_to_world("yumi_barrier_r")

        viewer["left_ee_barrier"].set_transform(lb.np)
        viewer["right_ee_barrier"].set_transform(rb.np)

        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            barriers=cbf_list,
        )
        configuration.integrate_inplace(velocity, dt)
        dist_ee = (
            np.linalg.norm(
                configuration.get_transform_frame_to_world(
                    "yumi_barrier_l"
                ).translation
                - configuration.get_transform_frame_to_world(
                    "yumi_barrier_r"
                ).translation
            )
            * 100
        )
        print(f"Distance between end effectors: {dist_ee :0.1f}cm")
        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
