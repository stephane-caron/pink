#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Universal Robots UR5 arm tracking a moving target."""

import meshcat_shapes
import numpy as np
import qpsolvers
from loop_rate_limiters import RateLimiter

import pink
from pink import solve_ik
from pink.barriers import BodySphericalCBF, ConfigurationCBF
from pink.tasks import FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, " "try ``pip install robot_descriptions``"
    ) from exc  # noqa: E501


if __name__ == "__main__":
    robot = load_robot_description("yumi_description", root_joint=None)

    viz = start_meshcat_visualizer(robot)

    O1 = np.array([0.05, 0.05, 0])
    O2 = np.array([-0.05, -0.05, 0])
    left_end_effector_task = FrameTask(
        "yumi_link_7_l",
        position_cost=50.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=100,  # tuned for this setup
    )
    right_end_effector_task = FrameTask(
        "yumi_link_7_r",
        position_cost=50.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=100,  # tuned for this setup
    )

    pos_cbf = BodySphericalCBF(
        ("yumi_link_7_l", "yumi_link_7_r"),
        O1=O1,
        O2=O2,
        d_min=0.2,
        gain=100.0,
        r=1.0,
    )

    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )

    configuration_cbf = ConfigurationCBF(robot.model, gain=1, r=400.0)
    cbf_list = [pos_cbf, configuration_cbf]
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
    meshcat_shapes.sphere(viewer["left_barrier"], opacity=0.4, color=0xFF0000, radius=0.1)
    meshcat_shapes.sphere(viewer["right_barrier"], opacity=0.4, color=0x00FF00, radius=0.1)
    meshcat_shapes.frame(viewer["right_end_effector"], opacity=1.0)
    meshcat_shapes.frame(viewer["left_end_effector_target"], opacity=1.0)
    meshcat_shapes.frame(viewer["right_end_effector_target"], opacity=1.0)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "ecos" in qpsolvers.available_solvers:
        solver = "ecos"

    rate = RateLimiter(frequency=150.0)
    dt = rate.period
    t = 0.0  # [s]
    l_y_des = np.array([0.392, 0.392, 0.6])
    r_y_des = np.array([0.392, 0.392, 0.6])
    l_dy_des = np.zeros(3)
    r_dy_des = np.zeros(3)

    while True:
        # Calculate desired trajectory
        A = 0.1
        B = 0.2
        # z -- 0.4 - 0.8
        l_y_des[:] = 0.6, 0.1 + B * np.sin(3 * t + np.pi / 4), 0.6 + A * np.sin(t)
        r_y_des[:] = 0.6, -0.1 - B * np.sin(t), 0.6 + A * np.sin(3 * t - np.pi / 4)
        l_dy_des[:] = 0, B * np.cos(t), A * np.cos(t)
        r_dy_des[:] = 0, -B * np.cos(t), A * np.cos(t)

        left_end_effector_task.transform_target_to_world.translation = l_y_des
        right_end_effector_task.transform_target_to_world.translation = r_y_des

        # Update visualization frames
        viewer["left_end_effector"].set_transform(
            configuration.get_transform_frame_to_world(left_end_effector_task.frame).np
        )
        viewer["right_end_effector"].set_transform(
            configuration.get_transform_frame_to_world(right_end_effector_task.frame).np
        )
        viewer["left_end_effector_target"].set_transform(left_end_effector_task.transform_target_to_world.np)
        viewer["right_end_effector_target"].set_transform(right_end_effector_task.transform_target_to_world.np)

        lb = configuration.get_transform_frame_to_world(left_end_effector_task.frame)
        lb.translation += O1

        rb = configuration.get_transform_frame_to_world(right_end_effector_task.frame)
        rb.translation += O2

        viewer["left_barrier"].set_transform(lb.np)
        viewer["right_barrier"].set_transform(rb.np)

        # Compute velocity and integrate it into next configuration
        # Note that default position limit handle given trajectory
        # much worse than CBF. Hence, we disable it here.
        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            cbfs=cbf_list,
            use_position_limit=False,
        )
        configuration.integrate_inplace(velocity, dt)
        dist = (
            np.linalg.norm(
                configuration.get_transform_frame_to_world("yumi_link_7_l").translation
                - configuration.get_transform_frame_to_world("yumi_link_7_r").translation
                + O1
                - O2
            )
            * 100
        )
        print(f"Distance between end effectors: {dist :0.1f}cm")  # noqa: E501
        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
