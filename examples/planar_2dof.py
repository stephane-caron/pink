#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# dependencies = ["daqp", "loop-rate-limiters", "viser", "pin-pink",
# "qpsolvers"]
# ///

"""Example of a planar 2-DOF robot arm moving in the XY-plane.

This example demonstrates inverse kinematics on a simple 2-DOF planar
manipulator. Both joints rotate around the Z-axis, constraining all motion
to the XY-plane. The end-effector follows a circular trajectory.
"""

import os

import numpy as np
import pinocchio as pin
import qpsolvers
from loop_rate_limiters import RateLimiter
from viser import GuiButtonHandle, GuiEvent, GuiSliderHandle
from viser_shapes import add_grid

import pink
from pink import solve_ik
from pink.tasks import ManipulabilityTask
from pink.visualization import start_viser_visualizer

if __name__ == "__main__":
    # Load robot description
    urdf_path = os.path.join(
        os.path.dirname(__file__),
        "robots",
        "planar_2dof.urdf",
    )
    robot = pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=["."],
        root_joint=None,
    )

    # Initialize visualization
    viz = start_viser_visualizer(robot, open=False)
    viewer = viz.viewer
    add_grid(viewer)

    # Define tasks
    manipulability_task = ManipulabilityTask(
        "end_effector",
        cost=1.0,
        lm_damping=1e-3,
        manipulability_rate=50.0,
        mask="planar_xy",
    )

    tasks = [
        manipulability_task,
    ]

    # Initialize configuration
    configuration = pink.Configuration(
        robot.model, robot.data, np.ones(robot.model.nq)
    )

    # Set task targets from initial configuration
    for task in tasks:
        if isinstance(task, ManipulabilityTask):
            continue  # Manipulability task does not have a target
        task.set_target_from_configuration(configuration)

    viz.display(configuration.q)
    reset_button = viewer.gui.add_button("Reset")
    lm_damping_slider = viewer.gui.add_slider(
        "Manipulability Damping",
        min=0.0,
        max=1e-2,
        initial_value=manipulability_task.lm_damping,
        step=1e-4,
    )
    manipulability_rate_slider = viewer.gui.add_slider(
        "Manipulability Rate",
        min=-100.0,
        max=100.0,
        initial_value=manipulability_task.manipulability_rate,
        step=1.0,
    )

    @reset_button.on_click
    def _(_: GuiEvent[GuiButtonHandle]):
        global configuration
        configuration.update(q=np.array([1.0, 1.0]))

    @lm_damping_slider.on_update
    def _(event: GuiEvent[GuiSliderHandle]):
        manipulability_task.lm_damping = event.target.value

    @manipulability_rate_slider.on_update
    def _(event: GuiEvent[GuiSliderHandle]):
        manipulability_task.manipulability_rate = event.target.value

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=100.0, warn=False)
    dt = rate.period

    print("Planar 2-DOF robot example started.")
    print("The end-effector follows a circular trajectory in the XY-plane.")
    print("Open the Viser URL in your browser to visualize.")

    while True:
        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
