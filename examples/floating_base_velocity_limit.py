#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
"""Clamp floating-base velocities with FloatingBaseVelocityLimit."""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import qpsolvers

import pink
from pink.limits import FloatingBaseVelocityLimit
from pink.tasks import FrameTask

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try `[conda|pip] install robot_descriptions`"
    ) from exc


def main() -> None:
    """Run a short IK loop where the base velocity remains bounded."""
    robot = load_robot_description(
        "upkie_description", root_joint=pin.JointModelFreeFlyer()
    )
    configuration = pink.Configuration(robot.model, robot.data, robot.q0)

    floating_limit = FloatingBaseVelocityLimit(
        model=robot.model,
        base_frame=None,
        max_linear_velocity=[0.3, 0.3, 0.2],  # [m] / [s]
        max_angular_velocity=[0.8, 0.8, 0.8],  # [rad] / [s]
    )
    configuration.model.floating_base_velocity_limit = floating_limit

    base_task = FrameTask(
        floating_limit.base_frame,
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
    )
    base_task.set_target_from_configuration(configuration)

    # Ask the base to move fast toward a far target. The limit will keep the
    # free-flyer velocity bounded no matter how aggressive the target is.
    transform = base_task.transform_target_to_world.copy()
    transform.translation += np.array([2.0, 0.0, 0.0])
    base_task.set_target(transform)

    dt = 0.1  # [s]
    solver = qpsolvers.available_solvers[0]
    root_joint = robot.model.joints[robot.model.getJointId("root_joint")]

    for step in range(10):
        velocity = pink.solve_ik(configuration, [base_task], dt, solver=solver)
        base_velocity = velocity[root_joint.idx_v : root_joint.idx_v + 6]
        angular = base_velocity[3:]
        linear = base_velocity[:3]
        print(
            f"step {step:02d} | "
            f"linear = {linear}  [m/s], "
            f"angular = {angular}  [rad/s]"
        )
        configuration.integrate_inplace(velocity, dt)


if __name__ == "__main__":
    main()
