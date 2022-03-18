#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Stéphane Caron
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test inverse kinematics.
"""

import os
import unittest

import numpy as np
import pinocchio as pin

from pink import apply_configuration
from pink import solve_ik
from pink.models import build_from_urdf
from pink.tasks import BodyTask


class TestSolveIK(unittest.TestCase):

    """
    Test fixture for the solve_ik function.
    """

    def setUp(self):
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        jvrc_description = os.path.join(models_dir, "jvrc_description")
        upkie_description = os.path.join(models_dir, "upkie_description")
        self.jvrc_description = jvrc_description
        self.upkie_description = upkie_description

    def test_no_task(self):
        """
        Raise an error when the robot body is not found.
        """
        robot = build_from_urdf(self.jvrc_description)
        tasks = []
        v = solve_ik(robot, tasks, dt=1e-3)
        self.assertTrue(np.allclose(v, np.zeros(robot.nv)))

    def test_single_task_fulfilled(self):
        robot = build_from_urdf(self.upkie_description)
        configuration = apply_configuration(robot, robot.q0)
        left_contact_task = BodyTask(
            "left_contact", position_cost=1.0, orientation_cost=1.0
        )
        left_contact_task.set_target(
            configuration.get_transform_body_to_world("left_contact")
        )
        velocity = solve_ik(configuration, [left_contact_task], dt=5e-3)
        self.assertTrue(np.allclose(velocity, 0.0))

    def test_single_task_convergence(self):
        robot = build_from_urdf(self.upkie_description)
        configuration = apply_configuration(robot, robot.q0)
        left_contact_task = BodyTask(
            "left_contact", position_cost=1.0, orientation_cost=1.0
        )
        transform_init_to_world = configuration.get_transform_body_to_world(
            "left_contact"
        )
        transform_target_to_init = pin.SE3(
            np.eye(3), np.array([0.0, 0.0, 0.1])
        )
        transform_target_to_world = (
            transform_init_to_world * transform_target_to_init
        )
        left_contact_task.set_target(transform_target_to_world)
        dt = 5e-3  # [s]
        velocity = solve_ik(configuration, [left_contact_task], dt)

        # Initially we are nowhere near the target and moving
        self.assertFalse(np.allclose(velocity, 0.0))
        self.assertAlmostEqual(
            np.linalg.norm(
                left_contact_task.compute_error_in_body(configuration)
            ),
            0.1,
        )
        self.assertFalse(
            configuration.get_transform_body_to_world("left_contact").isApprox(
                transform_target_to_world, prec=1e-4
            )
        )

        nb_steps = 2
        for _ in range(nb_steps):
            q = configuration.integrate(velocity, dt)
            configuration = apply_configuration(robot, q)
            velocity = solve_ik(configuration, [left_contact_task], dt)

        # After nb_steps we are at the target and not moving
        self.assertTrue(np.allclose(velocity, 0.0))
        self.assertAlmostEqual(
            np.linalg.norm(
                left_contact_task.compute_error_in_body(configuration)
            ),
            0.0,
        )
        self.assertTrue(
            configuration.get_transform_body_to_world("left_contact").isApprox(
                transform_target_to_world, prec=1e-8
            )
        )

    def test_three_tasks_fulfilled(self):
        """
        No motion when all targets are reached.
        """
        robot = build_from_urdf(self.jvrc_description)
        configuration = apply_configuration(robot, robot.q0)
        left_ankle_task = BodyTask(
            "l_ankle", position_cost=1.0, orientation_cost=3.0
        )
        right_ankle_task = BodyTask(
            "r_ankle", position_cost=1.0, orientation_cost=3.0
        )
        pelvis_task = BodyTask(
            "PELVIS_S", position_cost=1.0, orientation_cost=3.0
        )

        left_ankle_task.set_target(
            configuration.get_transform_body_to_world("l_ankle")
        )
        right_ankle_task.set_target(
            configuration.get_transform_body_to_world("r_ankle")
        )
        pelvis_task.set_target(
            configuration.get_transform_body_to_world("PELVIS_S")
        )

        tasks = [pelvis_task, left_ankle_task, right_ankle_task]
        velocity = solve_ik(configuration, tasks, dt=5e-3)
        self.assertTrue(np.allclose(velocity, 0.0))


if __name__ == "__main__":
    unittest.main()
