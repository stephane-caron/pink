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

import os
import unittest

import numpy as np

from pink.tasks import BodyTask
from pink.tasks import TargetNotSet

from .mock_configuration import MockConfiguration


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
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        jvrc_description = os.path.join(models_dir, "jvrc_description")
        self.jvrc_description = jvrc_description
        self.mock_configuration = MockConfiguration()

    def test_target_not_set(self):
        """
        Raise an exception when the target is not set.
        """
        jetpack_task = BodyTask(
            "jetpack", position_cost=1.0, orientation_cost=0.1
        )
        with self.assertRaises(TargetNotSet):
            jetpack_task.compute_task_dynamics(self.mock_configuration)

    def test_target_set_properly(self):
        """
        Return target properly once it's set.
        """
        jetpack_task = BodyTask(
            "jetpack", position_cost=1.0, orientation_cost=0.1
        )
        T = self.mock_configuration.get_transform_body_to_world("jetpack")
        jetpack_task.set_target(T)
        self.assertIsNotNone(jetpack_task.transform_target_to_world)
        if jetpack_task.transform_target_to_world is None:  # help mypy
            return
        self.assertTrue(
            np.allclose(
                T.homogeneous,
                jetpack_task.transform_target_to_world.homogeneous,
            )
        )

    def test_target_is_a_copy(self):
        """
        Target is saved as a copy, not a reference to the original.
        """
        tail_task = BodyTask("tail", position_cost=1.0, orientation_cost=0.1)
        target = self.mock_configuration.get_transform_body_to_world("tail")
        tail_task.set_target(target)
        y = target.translation[1]
        target.translation[1] += 12.0
        if tail_task.transform_target_to_world is None:  # help mypy
            return
        self.assertAlmostEqual(
            tail_task.transform_target_to_world.translation[1], y
        )
        self.assertNotAlmostEqual(
            tail_task.transform_target_to_world.translation[1],
            target.translation[1],
        )

    def test_zero_error_when_target_at_body(self):
        """
        Error is zero when the target and body are at the same location.
        """
        tail_task = BodyTask("tail", position_cost=1.0, orientation_cost=0.1)
        target = self.mock_configuration.get_transform_body_to_world("tail")
        tail_task.set_target(target)  # error == 0
        J, e = tail_task.compute_task_dynamics(self.mock_configuration)
        self.assertTrue(
            np.allclose(J, self.mock_configuration.get_body_jacobian("tail"))
        )
        self.assertLess(np.linalg.norm(e), 1e-10)

    def test_unit_cost_qp_objective(self):
        """
        A unit cost vector means the QP objective is exactly :math:`(H, c) =
        (J^T J, -e^T J)`.
        """
        shark_task = BodyTask("shark", position_cost=1.0, orientation_cost=0.1)
        target = self.mock_configuration.get_transform_body_to_world("shark")
        shark_task.set_target(target)
        self.mock_configuration.move_body_somewhere_else("shark")  # error > 0
        J, e = shark_task.compute_task_dynamics(self.mock_configuration)
        shark_task.set_position_cost(1.0)
        shark_task.set_orientation_cost(1.0)
        shark_task.lm_damping = 0.0
        H, c = shark_task.compute_qp_objective(self.mock_configuration)
        self.assertTrue(np.allclose(J.T @ J, H))
        self.assertTrue(np.allclose(-e.T @ J, c))

    def test_zero_costs_same_as_disabling_lines(self):
        """
        Setting a position or orientation cost to zero yields the same QP
        objective as disabling the corresponding Jacobian and error
        coordinates.
        """
        otter_task = BodyTask("otter", position_cost=1.0, orientation_cost=0.1)
        target = self.mock_configuration.get_transform_body_to_world("otter")
        otter_task.set_target(target)
        self.mock_configuration.move_body_somewhere_else("shark")  # error > 0
        J, e = otter_task.compute_task_dynamics(self.mock_configuration)
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
            otter_task.set_position_cost(position_cost)
            otter_task.set_orientation_cost(orientation_cost)
            otter_task.lm_damping = 0.0
            H, c = otter_task.compute_qp_objective(self.mock_configuration)
            H_check = J[indexes].T @ J[indexes]
            c_check = -e[indexes].T @ J[indexes]
            cost = qd.T @ H @ qd + c @ qd
            cost_check = qd.T @ H_check @ qd + c_check @ qd
            self.assertAlmostEqual(cost, cost_check)


if __name__ == "__main__":
    unittest.main()
