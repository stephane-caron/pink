#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

"""A pair of UR3 arms on a mobile body tracking moving targets."""

from typing import Tuple

import hppfcl as fcl
import numpy as np
import pinocchio as pin
import qpsolvers

import meshcat_shapes
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.utils import custom_configuration_vector
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
    ) from exc


def create_floating_base(
    name: str = "base",
    density: float = 1e-3,
    width: float = 0.8,
    height: float = 0.4,
    depth: float = 0.2,
) -> Tuple[pin.Model, pin.GeometryModel]:
    """Creating a floating base body to hold both arms.

    Args:
        name: Name of the floating base body.
        density: Mass density in kg / m³.
        width: Body width in meters.
        height: Body height in meters.
        depth: Body depth in meters.

    Returns:
        Joint and visual models.
    """
    model = pin.Model()
    mass = density * width * height * depth
    body_inertia = pin.Inertia.FromBox(mass, width, height, depth)
    parent = 0  # universe
    joint_id = model.addJoint(
        parent, pin.JointModelFreeFlyer(), pin.SE3.Identity(), name
    )
    model.appendBodyToJoint(joint_id, body_inertia, pin.SE3.Identity())
    model.addJointFrame(joint_id, -1)
    model.addBodyFrame(f"{name}_body", joint_id, pin.SE3.Identity(), -1)

    visual_model = pin.GeometryModel()
    box = fcl.Box(width, height, depth)
    box_placement = pin.SE3.Identity()
    box_placement.translation = np.array(
        [-0.0 * width, -0.0 * height, -0.5 * depth]
    )
    geom_obj = pin.GeometryObject("box", joint_id, box, box_placement)
    geom_obj.meshColor = np.array([0.9, 0.8, 0.6, 1.0])
    visual_model.addGeometryObject(geom_obj)
    return model, visual_model


def prefix_frames(
    model: pin.Model, visual_model: pin.GeometryModel, prefix: str
) -> None:
    for frame in model.frames:
        frame.name = f"{prefix}_{frame.name}"
    for i, name in enumerate(model.names):
        model.names[i] = f"{prefix}_{name}"
    for geom in visual_model.geometryObjects:
        geom.name = f"{prefix}_{geom.name}"


if __name__ == "__main__":
    model, visual_model = create_floating_base("base")

    left_arm = load_robot_description("ur3_description", root_joint=None)
    prefix_frames(left_arm.model, left_arm.visual_model, "left")
    left_arm_placement = pin.SE3.Identity()
    left_arm_placement.translation = np.array([0.3, 0.0, 0.0])
    model, visual_model = pin.appendModel(
        model,
        left_arm.model,
        visual_model,
        left_arm.visual_model,
        model.getFrameId("base"),
        left_arm_placement,
    )

    right_arm = load_robot_description("ur3_description", root_joint=None)
    prefix_frames(right_arm.model, right_arm.visual_model, "right")
    right_arm_placement = pin.SE3.Identity()
    right_arm_placement.translation = np.array([-0.3, 0.0, 0.0])
    model, visual_model = pin.appendModel(
        model,
        right_arm.model,
        visual_model,
        right_arm.visual_model,
        model.getFrameId("base"),
        right_arm_placement,
    )

    robot = pin.RobotWrapper(
        model,
        collision_model=visual_model,  # well...
        visual_model=visual_model,
    )

    # Initialize visualization
    viz = start_meshcat_visualizer(robot)
    viewer = viz.viewer
    meshcat_shapes.frame(viewer["base"], opacity=1.0)
    meshcat_shapes.frame(viewer["base_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["left_ee"], opacity=1.0)
    meshcat_shapes.frame(viewer["left_ee_target"], opacity=0.5)
    meshcat_shapes.frame(viewer["right_ee"], opacity=1.0)
    meshcat_shapes.frame(viewer["right_ee_target"], opacity=0.5)

    base_task = FrameTask(
        "base",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=0.1,
    )
    left_ee_task = FrameTask(
        "left_ee_link",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=1.0,  # tuned for this setup
    )
    right_ee_task = FrameTask(
        "right_ee_link",
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=1.0,  # tuned for this setup
    )
    posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )

    tasks = [base_task, left_ee_task, right_ee_task, posture_task]

    q_ref = custom_configuration_vector(
        robot,
        left_shoulder_lift_joint=1.0,
        left_shoulder_pan_joint=1.0,
        left_elbow_joint=1.0,
        right_shoulder_lift_joint=1.0,
        right_shoulder_pan_joint=1.0,
        right_elbow_joint=1.0,
    )
    configuration = pink.Configuration(robot.model, robot.data, q_ref)
    for task in tasks:
        task.set_target_from_configuration(configuration)
    viz.display(configuration.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]
    while True:
        # Update task targets
        base_target = base_task.transform_target_to_world
        base_target.translation[2] = 0.3 * np.sin(2.0 * t)
        left_ee_target = left_ee_task.transform_target_to_world
        left_ee_target.translation[1] = 0.5 + 0.1 * np.sin(2.0 * t)
        left_ee_target.translation[2] = 0.2
        right_ee_target = right_ee_task.transform_target_to_world
        right_ee_target.translation[1] = 0.5 + 0.1 * np.sin(2.0 * t)
        right_ee_target.translation[2] = 0.2

        # Update visualization frames
        viewer["base_target"].set_transform(base_target.np)
        viewer["base"].set_transform(
            configuration.get_transform_frame_to_world(base_task.frame).np
        )
        viewer["left_ee_target"].set_transform(left_ee_target.np)
        viewer["left_ee"].set_transform(
            configuration.get_transform_frame_to_world(left_ee_task.frame).np
        )
        viewer["right_ee_target"].set_transform(right_ee_target.np)
        viewer["right_ee"].set_transform(
            configuration.get_transform_frame_to_world(right_ee_task.frame).np
        )

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
