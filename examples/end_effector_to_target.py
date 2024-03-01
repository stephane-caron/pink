#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Move an end-effector to a prescribed target."""

import numpy as np
import pinocchio
from robot_descriptions.loaders.pinocchio import load_robot_description

import pink

robot = load_robot_description("ur10_description")

# Frame details
FRAME_NAME = "ee_frame"
joint_name = robot.model.names[-1]
parent_joint = robot.model.getJointId(joint_name)
parent_frame = robot.model.getFrameId(joint_name)
placement = pinocchio.SE3.Identity()

ee_frame = robot.model.addFrame(
    pinocchio.Frame(
        FRAME_NAME,
        parent_joint,
        parent_frame,
        placement,
        pinocchio.FrameType.OP_FRAME,
    )
)
robot.data = pinocchio.Data(robot.model)
low = robot.model.lowerPositionLimit
high = robot.model.upperPositionLimit
robot.q0 = pinocchio.neutral(robot.model)
# Task Details:
np.random.seed(0)

q_final = np.array(
    [
        np.random.uniform(low=low[i], high=high[i], size=(1,))[0]
        for i in range(6)
    ]
)
pinocchio.forwardKinematics(robot.model, robot.data, q_final)

target_pose = robot.data.oMi[parent_joint]
ee_task = pink.tasks.FrameTask(FRAME_NAME, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
ee_task.set_target(target_pose)
tasks = [ee_task]

# IK:
dt = 1e-2
damping = 1e-8
niter = 10000
solver = "quadprog"

pink_configuration = pink.Configuration(robot.model, robot.data, robot.q0)

for i in range(niter):
    dv = pink.solve_ik(
        pink_configuration,
        [ee_task],  # , config_task],
        dt=dt,
        damping=damping,
        solver=solver,
    )
    q_out = pinocchio.integrate(robot.model, pink_configuration.q, dv * dt)
    pink_configuration = pink.Configuration(robot.model, robot.data, q_out)
    pinocchio.updateFramePlacements(robot.model, robot.data)
    err = ee_task.compute_error(pink_configuration)
    print(i, err)
    if np.linalg.norm(ee_task.compute_error(pink_configuration)) < 1e-8:
        break
