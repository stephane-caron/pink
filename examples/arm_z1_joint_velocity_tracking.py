#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# dependencies = ["daqp", "loop-rate-limiters", "meshcat", "pin-pink",
# "qpsolvers", "robot_descriptions"]
# ///

"""Unitree Z1 arm tracking a reference joint-velocity trajectory."""

import numpy as np
import qpsolvers
from loop_rate_limiters import RateLimiter
from robot_descriptions.loaders.pinocchio import load_robot_description

import pink
from pink import solve_ik
from pink.tasks import JointVelocityTask
from pink.visualization import start_meshcat_visualizer

if __name__ == "__main__":
    print(
        "In this example, the arm tracks a sinusoidal joint-velocity "
        "trajectory that is unfeasible at times.\nThe trajectory is only "
        "tracked while the robot stays within joint limits."
    )
    robot = load_robot_description("z1_description")
    viz = start_meshcat_visualizer(robot)
    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    viz.display(configuration.q)

    # Our only task in this example is a joint-velocity task
    joint_velocity_task = JointVelocityTask(cost=1.0)

    # Select a QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        target_velocity = 2.0 * np.sin(t) * np.ones(robot.nv)
        joint_velocity_task.set_target(target_velocity, dt)
        velocity = solve_ik(
            configuration,
            tasks=[joint_velocity_task],
            dt=dt,
            solver=solver,
        )
        configuration.integrate_inplace(velocity, dt)
        viz.display(configuration.q)
        rate.sleep()
        t += dt
