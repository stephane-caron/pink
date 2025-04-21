#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Move the end-effector of a UR10 arm to a prescribed target."""

import numpy as np
import pinocchio
import qpsolvers

import pink
from pink.tasks import FrameTask

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try `[conda|pip] install robot_descriptions`"
    ) from exc

# IK parameters
dt = 1e-2
stop_thres = 1e-8

if __name__ == "__main__":
    robot = load_robot_description("ur10_description")

    # Frame details
    joint_name = robot.model.names[-1]
    parent_joint = robot.model.getJointId(joint_name)
    parent_frame = robot.model.getFrameId(joint_name)
    placement = pinocchio.SE3.Identity()

    FRAME_NAME = "ee_frame"
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

    # Task details
    np.random.seed(0)
    q_final = np.array(
        [
            np.random.uniform(low=low[i], high=high[i], size=(1,))[0]
            for i in range(6)
        ]
    )
    pinocchio.forwardKinematics(robot.model, robot.data, q_final)
    target_pose = robot.data.oMi[parent_joint]
    ee_task = FrameTask(FRAME_NAME, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    ee_task.set_target(target_pose)

    configuration = pink.Configuration(robot.model, robot.data, robot.q0)
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
        q_out = pinocchio.integrate(robot.model, configuration.q, dv * dt)
        configuration = pink.Configuration(robot.model, robot.data, q_out)
        pinocchio.updateFramePlacements(robot.model, robot.data)
        error_norm = np.linalg.norm(ee_task.compute_error(configuration))
        nb_steps += 1

    print(f"Terminated after {nb_steps} steps with {error_norm = :.2}")
