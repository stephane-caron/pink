#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# dependencies = [
#     "daqp",
#     "meshcat",
#     "pin-pink",
#     "qpsolvers",
#     "robot_descriptions",
# ]
# ///

"""Solve IK with the Piper arm end-effector at a prescribed target."""

import sys

import meshcat_shapes
import numpy as np
import pinocchio
import pinocchio as pin
import qpsolvers
from robot_descriptions.loaders.pinocchio import load_robot_description
from scipy.spatial.transform import Rotation

import pink
from pink.tasks import FrameTask
from pink.visualization import start_meshcat_visualizer

# IK parameters
dt = 1e-2
stop_thres = 1e-8

if __name__ == "__main__":
    robot = load_robot_description("piper_description")
    model = robot.model

    viz = start_meshcat_visualizer(robot)
    viewer = viz.viewer
    meshcat_shapes.frame(viewer["end_effector_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["end_effector"], opacity=1.0)

    # Frame details
    joint_name = model.names[-1]
    parent_joint = model.getJointId(joint_name)

    FRAME_NAME = "joint6"
    data = pin.Data(model)
    low = model.lowerPositionLimit
    high = model.upperPositionLimit
    q_init = pin.neutral(model)

    # Task details
    np.random.seed(0)
    q_final = np.array(
        [
            np.random.uniform(low=low[i], high=high[i], size=(1,))[0]
            for i in range(model.nq)
        ]
    )
    pin.forwardKinematics(model, data, q_final)
    target_pose = data.oMi[parent_joint]
    ee_task = FrameTask(FRAME_NAME, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])

    target = pin.SE3.Identity()
    hit_limit = "--hit-limit" in sys.argv
    target.translation = (
        np.array([0.1, 0.2, 0.3]) if hit_limit else np.array([0.0, 0.2, 0.6])
    )
    target.rotation = Rotation.from_euler("xyz", [0, 0, 0]).as_matrix()
    viewer["end_effector_target"].set_transform(target.np)
    ee_task.set_target(target)

    configuration = pink.Configuration(model, data, q_init)
    viz.display(configuration.q)
    error_norm = np.linalg.norm(ee_task.compute_error(configuration))
    print(f"Starting from {error_norm = :.2}")
    print(f"Desired precision is error_norm < {stop_thres}")

    nb_steps = 0
    while error_norm > stop_thres:
        dv = pink.solve_ik(
            configuration,
            tasks=[ee_task],
            dt=dt,
            damping=1e-8,
            solver=(
                "daqp"
                if "daqp" in qpsolvers.available_solvers
                else qpsolvers.available_solvers[0]
            ),
        )
        q_out = pin.integrate(model, configuration.q, dv * dt)
        q_out = np.clip(q_out, low, high)
        configuration = pink.Configuration(model, data, q_out)
        pin.updateFramePlacements(model, data)
        viewer["end_effector"].set_transform(
            configuration.get_transform_frame_to_world(ee_task.frame).np
        )
        viz.display(configuration.q)
        error_norm = np.linalg.norm(ee_task.compute_error(configuration))
        nb_steps += 1

    print(f"Terminated after {nb_steps} steps with {error_norm = :.2}")
