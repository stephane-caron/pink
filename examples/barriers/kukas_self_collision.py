#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Two iiwas iiw with full-body self-collision avoidance using hpp-fcl."""

import meshcat_shapes
import numpy as np
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter

import pink
from pink import solve_ik
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

    configuration = pink.Configuration(
        robot.model,
        robot.data,
        q_ref,
        collision_model=robot.collision_model,
        srdf_path=srdf_path,
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
        len(configuration.collision_model.collisionPairs),
        gain=10.0,
        safe_displacement_gain=1.0,
    )

    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )

    cbf_list = [collision_barrier]
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

    rate = RateLimiter(frequency=100.0)
    dt = rate.period
    t = 0.0  # [s]
    l_y_des = np.array([0.392, -0.392, 0.6])
    r_y_des = np.array([0.392, 0.392, 0.6])

    left_end_effector_task.transform_target_to_world.translation = l_y_des
    right_end_effector_task.transform_target_to_world.translation = r_y_des

    l_dy_des = np.zeros(3)
    r_dy_des = np.zeros(3)

    while True:
        # Calculate desired trajectory
        A = 0.1
        B = 0.1

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
            barriers=cbf_list,
        )
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
