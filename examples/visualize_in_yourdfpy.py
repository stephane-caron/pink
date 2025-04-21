#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Upkie wheeled biped bending its knees."""

import numpy as np
import pinocchio as pin
import qpsolvers
import yourdfpy

import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.utils import custom_configuration_vector

try:
    from loop_rate_limiters import RateLimiter
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples use loop rate limiters, "
        "try `[conda|pip] install loop-rate-limiters`"
    ) from exc

try:
    from robot_descriptions import upkie_description
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try `[conda|pip] install robot_descriptions`"
    )


if __name__ == "__main__":
    robot = load_robot_description(
        "upkie_description", root_joint=pin.JointModelFreeFlyer()
    )

    tasks = {
        "base": FrameTask(
            "base",
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=1.0,  # [cost] / [rad]
        ),
        "left_contact": FrameTask(
            "left_contact",
            position_cost=[0.1, 0.0, 0.1],  # [cost] / [m]
            orientation_cost=0.0,  # [cost] / [rad]
        ),
        "right_contact": FrameTask(
            "right_contact",
            position_cost=[0.1, 0.0, 0.1],  # [cost] / [m]
            orientation_cost=0.0,  # [cost] / [rad]
        ),
        "posture": PostureTask(
            cost=1e-3,  # [cost] / [rad]
        ),
    }

    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    for body, task in tasks.items():
        if type(task) is FrameTask:
            task.set_target_from_configuration(configuration)

    tasks["posture"].set_target(
        custom_configuration_vector(robot, left_knee=0.2, right_knee=-0.2)
    )

    left_contact_target = configuration.get_transform_frame_to_world(
        "left_contact"
    )
    right_contact_target = configuration.get_transform_frame_to_world(
        "right_contact"
    )

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    animation_time = 0.0  # [s]
    visualizer_fps = 100  # [Hz]
    rate = RateLimiter(frequency=visualizer_fps, warn=False)

    def callback(scene, dz=0.05):
        """Callback function for the visualizer."""
        global animation_time, configuration
        dt = rate.period

        # Update task targets
        t = animation_time
        left_contact_target.translation[2] += 0.1 * np.sin(t) * dt
        right_contact_target.translation[2] += 0.1 * np.sin(t) * dt
        tasks["left_contact"].set_target(left_contact_target)
        tasks["right_contact"].set_target(right_contact_target)

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks.values(), dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Display resulting configuration
        actuated_joints = configuration.q[7:]
        viz.update_cfg(actuated_joints)

        # Regulate visualizer FPS
        animation_time += dt
        rate.sleep()

    viz = yourdfpy.URDF.load(upkie_description.URDF_PATH)
    viz.show(callback=callback)
