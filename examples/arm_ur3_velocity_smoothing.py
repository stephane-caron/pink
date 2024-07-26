#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

"""UR3 arm tracking a moving target with velocity smoothing."""

import matplotlib.pyplot as plt
import meshcat_shapes
import numpy as np
import qpsolvers
from loop_rate_limiters import RateLimiter

import pink
from pink import solve_ik
from pink.tasks import FrameTask, LowAccelerationTask, PostureTask
from pink.utils import custom_configuration_vector
from pink.visualization import start_meshcat_visualizer

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Examples need the 'robot_descriptions' package "
        "(available from conda-forge or PyPI)"
    )


if __name__ == "__main__":
    robot = load_robot_description("ur3_description", root_joint=None)
    viz = start_meshcat_visualizer(robot)

    # Define inverse kinematics tasks
    end_effector_task = FrameTask(
        "ee_link",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=1.0,  # tuned for this setup
    )
    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )
    low_acceleration_task = LowAccelerationTask(
        cost=1e-1,  # [cost] * [s]^2 / [rad]
    )

    # Initial configuration and task setup
    q_ref = custom_configuration_vector(
        robot,
        shoulder_lift_joint=1.0,
        shoulder_pan_joint=1.0,
        elbow_joint=1.0,
    )
    configuration = pink.Configuration(robot.model, robot.data, q_ref)
    end_effector_task.set_target_from_configuration(configuration)
    posture_task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    viewer = viz.viewer
    meshcat_shapes.frame(viewer["end_effector_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["end_effector"], opacity=1.0)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "quadprog" in qpsolvers.available_solvers:
        solver = "quadprog"

    rate = RateLimiter(frequency=200.0)
    dt = rate.period
    configurations, velocities, times = [], [], []
    nb_steps = 3000
    t = 0.0  # [s]
    for step in range(nb_steps):
        # Update task targets
        end_effector_target = end_effector_task.transform_target_to_world
        end_effector_target.translation[1] = 0.5 + 0.1 * np.sin(2.0 * t)
        end_effector_target.translation[2] = 0.2

        # Update visualization frames
        viewer["end_effector_target"].set_transform(end_effector_target.np)
        viewer["end_effector"].set_transform(
            configuration.get_transform_frame_to_world(
                end_effector_task.frame
            ).np
        )

        # Compute velocity and integrate it into next configuration
        tasks = (
            (end_effector_task, posture_task)
            if step < nb_steps // 2
            else (end_effector_task, low_acceleration_task)
        )
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)
        low_acceleration_task.set_previous_velocity(velocity)

        # Append plotting data to lists
        configurations.append(configuration.q)
        times.append(t)
        velocities.append(velocity)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt

    # Plot velocities and configurations over time
    velocities = np.array(velocities)
    configurations = np.array(configurations)
    plt.figure()
    plt.plot(times, velocities)
    plt.plot(
        [times[nb_steps // 2]] * 2,
        [velocities.min(), velocities.max()],
        "k--",
    )
    plt.title("Joint velocities (posture then low-accel task)")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint velocity [rad/s]")
    plt.figure()
    plt.plot(times, configurations)
    plt.plot(
        [times[nb_steps // 2]] * 2,
        [configurations.min(), configurations.max()],
        "k--",
    )
    plt.title("Configuration data")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint position [rad]")
    plt.show()
