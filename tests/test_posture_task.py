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

import os
import unittest

import numpy as np

from pink.tasks import PostureTask
from pink.tasks import TargetNotSet

from .mock_configuration import MockConfiguration


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
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        jvrc_description = os.path.join(models_dir, "jvrc_description")
        self.jvrc_description = jvrc_description
        self.mock_configuration = MockConfiguration()

    def test_target_not_set(self):
        """
        Raise an exception when the target is not set.
        """
        task = PostureTask(cost=1.0)
        with self.assertRaises(TargetNotSet):
            task.compute_task_dynamics(self.mock_configuration)

    def test_target_set_properly(self):
        """
        Return target properly once it's set.
        """
        task = PostureTask(cost=1.0)
        task.set_target(self.mock_configuration.q)
        self.assertIsNotNone(task.target_q)
        if task.target_q is None:  # help mypy
            return
        self.assertTrue(np.allclose(self.mock_configuration.q, task.target_q))

    def test_target_is_a_copy(self):
        """
        Target is saved as a copy, not a reference to the original.
        """
        task = PostureTask(cost=1.0)
        q = self.mock_configuration.q
        task.set_target(q)
        y = q[2]
        q[2] += 12.0
        if task.target_q is None:  # help mypy
            return
        self.assertAlmostEqual(task.target_q[2], y)
        self.assertNotAlmostEqual(task.target_q[2], q[2])

    def test_zero_error_when_target_at_body(self):
        """
        Error is zero when the target and body are at the same location.
        """
        task = PostureTask(cost=1.0)
        q = self.mock_configuration.q
        task.set_target(q)  # error == 0
        _, e = task.compute_task_dynamics(self.mock_configuration)
        self.assertLess(np.linalg.norm(e), 1e-10)

    def test_unit_cost_qp_objective(self):
        """
        A unit cost vector means the QP objective is exactly :math:`(H, c) =
        (J^T J, -e^T J)`.
        """
        task = PostureTask(cost=1.0)
        q = self.mock_configuration.q
        task.set_target(q)
        self.mock_configuration.q[1] += 1.0
        self.mock_configuration.q[3] += 1.0
        self.mock_configuration.q[5] += 1.0
        J, e = task.compute_task_dynamics(self.mock_configuration)
        H, c = task.compute_qp_objective(self.mock_configuration)
        self.assertTrue(np.allclose(J.T @ J, H))
        self.assertTrue(np.allclose(-e.T @ J, c))

    def test_zero_cost_same_as_disabling_task(self):
        """
        The task has no effect when its cost is zero.
        """
        task = PostureTask(cost=0.0)
        q = self.mock_configuration.q
        task.set_target(q)
        J, e = task.compute_task_dynamics(self.mock_configuration)
        H, c = task.compute_qp_objective(self.mock_configuration)
        qd = np.random.random(J.shape[1:])
        cost = qd.T @ H @ qd + c @ qd
        self.assertAlmostEqual(cost, 0.0)


if __name__ == "__main__":
    unittest.main()
