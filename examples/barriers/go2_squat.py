#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""GO2 squat with z-axis barrier."""

import meshcat_shapes
import numpy as np
import qpsolvers
from loop_rate_limiters import RateLimiter
import pinocchio as pin
from time import perf_counter
import pink
from pink import solve_ik
from pink.barriers import PositionBarrier
from pink.tasks import FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try ``pip install robot_descriptions``"
    ) from exc  # noqa: E501


if __name__ == "__main__":
    robot = load_robot_description(
        "go2_description", root_joint=pin.JointModelFreeFlyer())
    viz = start_meshcat_visualizer(robot)
    
    q_ref = np.array([-0.,  0.,  0.3,  0.,  0., 0.0,  1.,
                      0.0,  0.8, -1.57,
                      0.0,  0.8, -1.57,
                      0.0,  0.8, -1.57,
                      0.0,  0.8, -1.57])
    
    configuration = pink.Configuration(
        robot.model, robot.data, q_ref)

    base_task = FrameTask(
        "base",
        position_cost=50.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
    )

    posture_task = PostureTask(
        cost=1e-5,  # [cost] / [rad]
    )

    pos_barrier = PositionBarrier(
        "base",
        indices=[1,2],
        p_max=np.array([0, 0.35]),
        gain=np.array([100.0, 100.0]),
        r=1.0,
    )
    barriers_list = [pos_barrier]

    tasks = [base_task, posture_task]

    for foot in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]:
        task = FrameTask(
            foot,
            position_cost=200.0,  # [cost] / [m]
            orientation_cost=0.0,  # [cost] / [rad]
        )
        tasks.append(task)

    for task in tasks:
        task.set_target_from_configuration(configuration)


    viewer = viz.viewer
    opacity = 0.5  # Set the desired opacity level (0 transparent, 1 opaque)

    meshcat_shapes.frame(viewer["base_target"], opacity=1.0)
    meshcat_shapes.frame(viewer["base"], opacity=1.0)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "osqp" in qpsolvers.available_solvers:
        solver = "osqp"

    rate = RateLimiter(frequency=200.0)
    dt = rate.period
    t = 0.0  # [s]
    period = 4
    omega = 2 * np.pi / period
    while True:
        # Update task targets
        end_effector_target = base_task.transform_target_to_world
        phase = (t//period)%2
        Ay = 0.1*(1-phase)
        Az = 0.2*phase
            
        end_effector_target.translation[1] = 0. + Ay * np.sin(omega*t)
        end_effector_target.translation[2] = 0.3 + Az * np.sin(omega*t)
        
        
        # Update visualization frames
        viewer["base_target"].set_transform(end_effector_target.np)
        viewer["base"].set_transform(
            configuration.get_transform_frame_to_world(
                base_task.frame
            ).np
        )

        # Compute velocity and integrate it into next configuration
        # Note that default position limit handle given trajectory
        # much worse than CBF. Hence, we disable it here.

        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            barriers=barriers_list,
        )
        configuration.integrate_inplace(velocity, dt)


        viz.display(configuration.q)
        rate.sleep()
        t += dt
