#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# dependencies = ["daqp", "loop-rate-limiters", "viser", "pin-pink",
# "qpsolvers", "robot_descriptions", "trimesh"]
# ///

"""JVRC-1 humanoid backbending by regulating its CoM position."""

import numpy as np
import pinocchio as pin
import qpsolvers
import trimesh
import viser
from loop_rate_limiters import RateLimiter
from robot_descriptions.loaders.pinocchio import load_robot_description

import pink
from pink import solve_ik
from pink.tasks import ComTask, FrameTask
from pink.visualization import start_viser_visualizer

if __name__ == "__main__":
    robot = load_robot_description(
        "jvrc_description", root_joint=pin.JointModelFreeFlyer()
    )

    # Initialize visualization
    viz = start_viser_visualizer(robot)
    viewer = viz.viewer

    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
    viz.display(configuration.q)

    # Initialize IK tasks
    left_ankle_task = FrameTask(
        "l_ankle", position_cost=1.0, orientation_cost=3.0
    )
    right_ankle_task = FrameTask(
        "r_ankle", position_cost=1.0, orientation_cost=3.0
    )
    com_task = ComTask(cost=2.0)
    tasks = [com_task, left_ankle_task, right_ankle_task]

    # IK: Set ankle task targets
    left_ankle_task.set_target(
        configuration.get_transform_frame_to_world("l_ankle")
        * pin.SE3(np.eye(3), np.array([0.1, 0.0, 0.0]))
    )
    right_ankle_task.set_target(
        configuration.get_transform_frame_to_world("r_ankle")
        * pin.SE3(np.eye(3), np.array([-0.1, 0.0, 0.0]))
    )

    # IK: Set CoM task target
    initial_com = pin.centerOfMass(robot.model, robot.data, configuration.q)
    desired_com = initial_com.copy()
    desired_com[2] += 0.05
    com_task.set_target(desired_com)

    # Visualization: add current and target CoM handles
    sphere = trimesh.creation.icosphere(radius=0.03)
    com_target_handle = viewer.scene.add_mesh_simple(
        "com_target",
        sphere.vertices,
        sphere.faces,
        color=(0.0, 1.0, 0.0),  # green
    )
    com_target_handle.position = tuple(desired_com.tolist())
    com_handle = viewer.scene.add_mesh_simple(
        "com",
        sphere.vertices,
        sphere.faces,
        color=(1.0, 0.0, 0.0),  # red
    )
    com_handle.position = tuple(initial_com.tolist())

    # Visualization: add ground plane
    left_ankle_target = left_ankle_task.transform_target_to_world.translation
    viewer.scene.set_background_image(np.full((1, 1, 3), 220, dtype=np.uint8))
    viewer.scene.add_grid(
        "/grid",
        width=2.0,
        height=2.0,
        position=(0.0, 0.0, left_ankle_target[2] - 0.11),
        plane_opacity=0.5,
    )

    # Interactive handle to drag the CoM target
    transform_handle = viewer.scene.add_transform_controls(
        "/com_target_control",
        position=tuple(desired_com.tolist()),
        wxyz=(1.0, 0.0, 0.0, 0.0),
        scale=0.2,
    )

    @transform_handle.on_update
    def _(handle: viser.TransformControlsEvent) -> None:
        new_target = np.array(handle.target.position)
        com_task.set_target(new_target)
        com_target_handle.position = handle.target.position

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period

    while True:
        pin.centerOfMass(robot.model, robot.data, configuration.q)
        com = robot.data.com[0]

        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            safety_break=False,
        )
        configuration.integrate_inplace(velocity, dt)
        viz.display(configuration.q)
        com_handle.position = tuple(com.tolist())

        rate.sleep()
