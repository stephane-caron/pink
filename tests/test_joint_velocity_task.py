#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Test fixture for the joint velocity task."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.tasks import JointVelocityTask


class TestJointVelocityTask(unittest.TestCase):
    """Test consistency of the joint velocity task.

    Note:
        This fixture only tests the task itself. Integration tests with the IK
        are carried out in :class:`TestSolveIK`.
    """

    def setUp(self):
        """Prepare test fixture."""
        robot = load_robot_description(
            "ur3_description",
            root_joint=pin.JointModelFreeFlyer(),
        )
        self.configuration = Configuration(robot.model, robot.data, robot.q0)

    def test_task_repr(self):
        """String representation reports the task gain, costs and target."""
        task = JointVelocityTask(cost=1.0)
        self.assertTrue("cost=" in repr(task))
        self.assertTrue("gain=" in repr(task))

    def test_unit_cost_qp_objective(self):
        """A unit cost vector means the QP objective is (J^T J, -e^T J)."""
        task = JointVelocityTask(cost=1.0)
        e = task.compute_error(self.configuration)
        J = task.compute_jacobian(self.configuration)
        H, c = task.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(J.T @ J, H))
        self.assertTrue(np.allclose(-e.T @ J, c))
