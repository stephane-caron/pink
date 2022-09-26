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
Test fixture for the posture task.
"""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

import pink
from pink.tasks import PostureTask, TargetNotSet


class TestPostureTask(unittest.TestCase):

    """
    Test consistency of the posture task.

    Note:
        This fixture only tests the task itself. Integration tests with the IK
        are carried out in :class:`TestSolveIK`.
    """

    def setUp(self):
        """
        Prepare test fixture.
        """
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        self.configuration = pink.apply_configuration(robot, robot.q0)

    def test_task_repr(self):
        """
        String representation reports the task gain, costs and target.
        """
        task = PostureTask(cost=1.0)
        self.assertTrue("cost=" in repr(task))
        self.assertTrue("gain=" in repr(task))

    def test_target_not_set(self):
        """
        Raise an exception when the target is not set.
        """
        task = PostureTask(cost=1.0)
        with self.assertRaises(TargetNotSet):
            task.compute_task_dynamics(self.configuration)

    def test_target_set_properly(self):
        """
        Return target properly once it's set.
        """
        task = PostureTask(cost=1.0)
        task.set_target(self.configuration.q)
        self.assertIsNotNone(task.target_q)
        if task.target_q is None:  # help mypy
            return
        self.assertTrue(np.allclose(self.configuration.q, task.target_q))

    def test_target_is_a_copy(self):
        """
        Target is saved as a copy, not a reference to the original.
        """
        task = PostureTask(cost=1.0)
        q = self.configuration.q
        with self.assertRaises(ValueError):
            q[2] += 12.0  # assignment destination is read-only
        task.set_target(q)
        self.assertIsNotNone(task.target_q)

    def test_zero_error_when_target_at_body(self):
        """
        Error is zero when the target and body are at the same location.
        """
        task = PostureTask(cost=1.0)
        task.set_target(self.configuration.q)  # error == 0
        _, e = task.compute_task_dynamics(self.configuration)
        self.assertLess(np.linalg.norm(e), 1e-10)

    def test_unit_cost_qp_objective(self):
        """
        A unit cost vector means the QP objective is exactly :math:`(H, c) =
        (J^T J, -e^T J)`.
        """
        task = PostureTask(cost=1.0)
        task.set_target(self.configuration.q)
        q_new = self.configuration.q.copy()
        q_new[1] += 1.0
        q_new[3] += 1.0
        q_new[5] += 1.0
        self.configuration.q = q_new
        J, e = task.compute_task_dynamics(self.configuration)
        H, c = task.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(J.T @ J, H))
        self.assertTrue(np.allclose(-e.T @ J, c))

    def test_zero_cost_same_as_disabling_task(self):
        """
        The task has no effect when its cost is zero.
        """
        task = PostureTask(cost=0.0)
        q = self.configuration.q
        task.set_target(q)
        J, e = task.compute_task_dynamics(self.configuration)
        H, c = task.compute_qp_objective(self.configuration)
        qd = np.random.random(J.shape[1:])
        cost = qd.T @ H @ qd + c @ qd
        self.assertAlmostEqual(cost, 0.0)
