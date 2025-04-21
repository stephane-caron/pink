#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Ivan Domrachev, Simeon Nedelchev

"""Universal Robots UR5 arm tracking a moving target."""

import argparse

import numpy as np
import qpsolvers

import meshcat_shapes
import pink
from pink import solve_ik
from pink.barriers import PositionBarrier
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        "-v",
        help="print out task errors and CBF values during execution",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    robot = load_robot_description("ur5_description", root_joint=None)
    viz = start_meshcat_visualizer(robot)

    end_effector_task = FrameTask(
        "ee_link",
        position_cost=10.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
    )

    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )

    pos_barrier = PositionBarrier(
        "ee_link",
        indices=[1],
        p_min=np.array([-0.4]),
        p_max=np.array([0.6]),
        gain=np.array([100.0]),
        safe_displacement_gain=1.0,
    )
    barriers = [pos_barrier]

    tasks = [end_effector_task, posture_task]

    q_ref = np.array(
        [
            1.27153374,
            -0.87988708,
            1.89104795,
            1.73996951,
            -0.24610945,
            -0.74979019,
        ]
    )
    configuration = pink.Configuration(robot.model, robot.data, q_ref)
    for task in tasks:
        task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    viewer = viz.viewer
    meshcat_shapes.frame(viewer["end_effector_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["end_effector"], opacity=1.0)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "osqp" in qpsolvers.available_solvers:
        solver = "osqp"

    rate = RateLimiter(frequency=200.0)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update task targets
        end_effector_target = end_effector_task.transform_target_to_world
        end_effector_target.translation[1] = 0.0 + 0.65 * np.sin(t / 4)
        end_effector_target.translation[2] = 0.5

        # Update visualization frames
        viewer["end_effector_target"].set_transform(end_effector_target.np)
        viewer["end_effector"].set_transform(
            configuration.get_transform_frame_to_world(
                end_effector_task.frame
            ).np
        )

        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            barriers=barriers,
        )
        configuration.integrate_inplace(velocity, dt)

        G, h = pos_barrier.compute_qp_inequalities(configuration, dt=dt)
        distance_to_manipulator = configuration.get_transform_frame_to_world(
            "ee_link"
        ).translation[1]
        if args.verbose:
            print(
                f"Task error: {end_effector_task.compute_error(configuration)}"
            )
            print(
                "Position CBF value: "
                f"{pos_barrier.compute_barrier(configuration)[0]:0.3f} >= 0"
            )
            print(f"Distance to manipulator: {distance_to_manipulator} <= 0.6")
            print("-" * 60)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
