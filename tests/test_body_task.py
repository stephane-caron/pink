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
Test fixture for the body task.
"""

import unittest

import numpy as np
import pinocchio as pin
from qpsolvers import solve_qp
from robot_descriptions.loaders.pinocchio import load_robot_description

import pink
from pink.tasks import BodyTask, TargetNotSet


class TestBodyTask(unittest.TestCase):

    """
    Test consistency of the body task.

    Note:
        This fixture only tests the task itself. Integration tests with the IK
        are carried out in :class:`TestSolveIK`.
    """

    def setUp(self):
        """
        Prepare test fixture.
        """
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        self.configuration = pink.apply_configuration(robot, robot.q0)

    def test_task_repr(self):
        """
        String representation reports the task gain, costs and target.
        """
        earflap_task = BodyTask(
            "earflap", position_cost=1.0, orientation_cost=0.1
        )
        self.assertTrue("gain=" in repr(earflap_task))
        self.assertTrue("cost=" in repr(earflap_task))
        self.assertTrue("target=" in repr(earflap_task))

    def test_target_not_set(self):
        """
        Raise an exception when the target is not set.
        """
        task = BodyTask("l_ankle", position_cost=1.0, orientation_cost=0.1)
        with self.assertRaises(TargetNotSet):
            task.compute_task_dynamics(self.configuration)

    def test_target_set_properly(self):
        """
        Return target properly once it's set.
        """
        task = BodyTask("l_ankle", position_cost=1.0, orientation_cost=0.1)
        T = self.configuration.get_transform_body_to_world("l_ankle")
        task.set_target(T)
        self.assertIsNotNone(task.transform_target_to_world)
        if task.transform_target_to_world is not None:  # help mypy
            self.assertTrue(
                np.allclose(
                    T.homogeneous,
                    task.transform_target_to_world.homogeneous,
                )
            )

    def test_target_is_a_copy(self):
        """
        Target is saved as a copy, not a reference to the original.
        """
        task = BodyTask("l_ankle", position_cost=1.0, orientation_cost=0.1)
        target = self.configuration.get_transform_body_to_world("l_ankle")
        task.set_target(target)
        y = target.translation[1]
        target.translation[1] += 12.0
        if task.transform_target_to_world is not None:  # help mypy
            self.assertAlmostEqual(
                task.transform_target_to_world.translation[1], y
            )
            self.assertNotAlmostEqual(
                task.transform_target_to_world.translation[1],
                target.translation[1],
            )

    def test_zero_error_when_target_at_body(self):
        """
        Error is zero when the target and body are at the same location.
        """
        task = BodyTask("r_ankle", position_cost=1.0, orientation_cost=0.1)
        target = self.configuration.get_transform_body_to_world("r_ankle")
        task.set_target(target)  # error == 0
        J, e = task.compute_task_dynamics(self.configuration)
        self.assertTrue(
            np.allclose(J, self.configuration.get_body_jacobian("r_ankle"))
        )
        self.assertLess(np.linalg.norm(e), 1e-10)

    def test_unit_cost_qp_objective(self):
        """
        A unit cost vector means the QP objective is exactly :math:`(H, c) =
        (J^T J, -e^T J)`.
        """
        task = BodyTask("r_wrist", position_cost=1.0, orientation_cost=0.1)
        transform_target_to_body = pin.SE3(
            np.eye(3), np.array([0.0, 0.01, 0.0])
        )
        target = (
            self.configuration.get_transform_body_to_world("r_wrist")
            * transform_target_to_body
        )
        task.set_target(target)
        J, e = task.compute_task_dynamics(self.configuration)
        task.set_position_cost(1.0)
        task.set_orientation_cost(1.0)
        task.lm_damping = 0.0
        H, c = task.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(J.T @ J, H))
        self.assertTrue(np.allclose(-e.T @ J, c))

    def test_zero_costs_same_as_disabling_lines(self):
        """
        Setting a position or orientation cost to zero yields the same QP
        objective as disabling the corresponding Jacobian and error
        coordinates.
        """
        task = BodyTask("l_wrist", position_cost=1.0, orientation_cost=0.1)
        transform_target_to_body = pin.SE3(
            np.eye(3), np.array([0.1, 0.02, 0.01])
        )
        target = (
            self.configuration.get_transform_body_to_world("l_wrist")
            * transform_target_to_body
        )
        task.set_target(target)
        J, e = task.compute_task_dynamics(self.configuration)
        qd = np.random.random(J.shape[1:])
        test_cases = {
            "position_only": (1.0, 0.0, slice(0, 3)),
            "orientation_only": (0.0, 1.0, slice(3, 6)),
            "position_0": ([1.0, 0.0, 0.0], 0.0, slice(0, 1)),
            "position_1": ([0.0, 1.0, 0.0], 0.0, slice(1, 2)),
            "position_2": ([0.0, 0.0, 1.0], 0.0, slice(2, 3)),
            "orientation_0": (0.0, [1.0, 0.0, 0.0], slice(3, 4)),
            "orientation_1": (0.0, [0.0, 1.0, 0.0], slice(4, 5)),
            "orientation_2": (0.0, [0.0, 0.0, 1.0], slice(5, 6)),
        }
        for case in test_cases.values():
            position_cost, orientation_cost, indexes = case
            task.set_position_cost(position_cost)
            task.set_orientation_cost(orientation_cost)
            task.lm_damping = 0.0
            H, c = task.compute_qp_objective(self.configuration)
            H_check = J[indexes].T @ J[indexes]
            c_check = -e[indexes].T @ J[indexes]
            cost = qd.T @ H @ qd + c @ qd
            cost_check = qd.T @ H_check @ qd + c_check @ qd
            self.assertAlmostEqual(cost, cost_check)

    def test_lm_damping_has_no_effect_at_target(self):
        """
        Levenberg-Marquardt damping has no effect when the task error is zero.
        """
        task = BodyTask("l_wrist", position_cost=1.0, orientation_cost=0.1)
        target = self.configuration.get_transform_body_to_world("l_wrist")
        task.set_target(target)
        task.lm_damping = 1e-8
        H_1, c_1 = task.compute_qp_objective(self.configuration)
        task.lm_damping = 1e-4
        H_2, c_2 = task.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(H_1, H_2))
        self.assertTrue(np.allclose(c_1, c_2))

    def test_lm_damping_has_effect_under_error(self):
        """
        Levenberg-Marquardt damping is indeed a damping: unless the task is
        fulfilled, it reduces velocities.
        """
        task = BodyTask("r_wrist", position_cost=1.0, orientation_cost=0.1)
        transform_target_to_body = pin.SE3(
            np.eye(3), np.array([0.0, 2.0, 0.0])
        )
        target = (
            self.configuration.get_transform_body_to_world("r_wrist")
            * transform_target_to_body
        )
        task.set_target(target)
        task.lm_damping = 1e-8
        H_1, c_1 = task.compute_qp_objective(self.configuration)
        task.lm_damping = 1e-4
        H_2, c_2 = task.compute_qp_objective(self.configuration)
        qd_1 = solve_qp(H_1, c_1, solver="quadprog")  # H_1 p.s.d. (LM damping)
        qd_2 = solve_qp(H_2, c_2, solver="quadprog")  # idem for H_2
        self.assertGreater(np.linalg.norm(qd_2 - qd_1), 1e-6)
