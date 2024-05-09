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
from pink.barriers import SelfCollisionBarrier, ConfigurationCBF
from pink.tasks import FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, " "try ``pip install robot_descriptions``"
    ) from exc  # noqa: E501


if __name__ == "__main__":
    robot = load_robot_description("baxter_description", root_joint=None)

    viz = start_meshcat_visualizer(robot)

    left_end_effector_task = FrameTask(
        "left_wrist",
        position_cost=50.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=100,  # tuned for this setup
    )
    right_end_effector_task = FrameTask(
        "right_wrist",
        position_cost=50.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=100,  # tuned for this setup
    )

    pos_cbf = SelfCollisionBarrier(
        robot,
        # exclude_pairs=[
        #     (
        #         f"yumi_link_{i}_{side}",
        #         f"yumi_link_{i+1}_{side}",
        #     )
        #     for i in range(6)
        #     for side in ["l", "r"]
        # ],
        gain=100.0,
        r=1.0,
    )

    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )

    configuration_cbf = ConfigurationCBF(robot.model, gain=1, r=100.0)
    cbf_list = [pos_cbf, configuration_cbf]
    tasks = [left_end_effector_task, right_end_effector_task, posture_task]

    q_ref = np.zeros(robot.nq)
    configuration = pink.Configuration(robot.model, robot.data, q_ref)
    for task in tasks:
        task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    viewer = viz.viewer
    meshcat_shapes.frame(viewer["left_end_effector"], opacity=1.0)
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
        l_y_des[:] = 0.6, 0.1 + B * np.sin(t), 0.6 + A * np.sin(t)
        r_y_des[:] = 0.6, -0.1 - B * np.sin(t), 0.6 + A * np.sin(t)
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
        viewer["left_end_effector_target"].set_transform(l_y_des)
        viewer["right_end_effector_target"].set_transform(r_y_des)

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
        print(
            f"Distance between end effectors: {np.linalg.norm(configuration.get_transform_frame_to_world('yumi_link_7_l').translation - configuration.get_transform_frame_to_world('yumi_link_7_r').translation)*100 :0.1f}cm"  # noqa: E501
        )
        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
