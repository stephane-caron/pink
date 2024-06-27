#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Ivan Domrachev, Simeon Nedelchev

"""Two iiwa14-s with full-body self-collision avoidance using hpp-fcl."""

import meshcat_shapes
import numpy as np
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter

import pink
from pink import solve_ik
from pink.utils import process_collision_pairs
from pink.barriers import SelfCollisionBarrier
from pink.tasks import FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer

if __name__ == "__main__":
    # Load the robot and define the custom frames
    urdf_path = "examples/barriers/models/iiwa14_spheres_collision.urdf"
    srdf_path = "examples/barriers/models/iiwa14_spheres_collision.srdf"
    robot = pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=["."],
        root_joint=None,
    )

    viz = start_meshcat_visualizer(robot)
    q_ref = np.zeros(robot.model.nq)

    # Collisions: processing collisions from urdf (include all) and srdf (exclude specified)
    # and updating collision model and creating corresponding collision data
    robot.collision_data = process_collision_pairs(robot.model, robot.collision_model, srdf_path)

    configuration = pink.Configuration(
        robot.model,
        robot.data,
        q_ref,
        collision_model=robot.collision_model,  # Collision model is required for self_collision_barrier
        collision_data=robot.collision_data,
    )

    # Pink tasks
    left_end_effector_task = FrameTask(
        "first_iiwa_link_7",
        position_cost=50.0,  # [cost] / [m]
        orientation_cost=10.0,  # [cost] / [rad]
    )
    right_end_effector_task = FrameTask(
        "second_iiwa_link_7",
        position_cost=50.0,  # [cost] / [m]
        orientation_cost=10.0,  # [cost] / [rad]
    )

    # Pink barriers
    collision_barrier = SelfCollisionBarrier(
        n_collision_pairs=len(robot.collision_model.collisionPairs),
        gain=20.0,
        safe_displacement_gain=1.0,
        d_min=0.05,
    )

    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )

    barriers = [collision_barrier]
    tasks = [left_end_effector_task, right_end_effector_task, posture_task]

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

    rate = RateLimiter(frequency=50.0)
    dt = rate.period
    t = 0.0  # [s]
    l_y_des = np.array([0.392, -0.392, 0.6])
    r_y_des = np.array([0.392, 0.392, 0.6])

    A = l_y_des.copy()
    B = r_y_des.copy()

    l_dy_des = np.zeros(3)
    r_dy_des = np.zeros(3)

    while True:
        # Make a sinusoidal trajectory between points A and B
        mu = (1 + np.cos(t)) / 2
        l_y_des[:] = A + (B - A + 0.2 * np.array([0, 0, np.sin(mu * np.pi) ** 2])) * mu
        r_y_des[:] = B + (A - B + 0.2 * np.array([0, 0, -np.sin(mu * np.pi) ** 2])) * mu

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

        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            barriers=barriers,
        )
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
