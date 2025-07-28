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
from pink.exceptions import TargetNotSet, TaskDefinitionError
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
        configuration = Configuration(robot.model, robot.data, robot.q0)
        self.configuration = configuration
        self.dt = 3e-3  # seconds
        self.nq = configuration.model.nq
        self.nv = configuration.model.nv

    def test_task_repr(self):
        """String representation reports the task gain, costs and target."""
        task = JointVelocityTask(cost=1.0)
        self.assertTrue("cost=" in repr(task))

    def test_target_not_set(self):
        """Exception raised when the target is unset."""
        task = JointVelocityTask(cost=1.0)
        with self.assertRaises(TargetNotSet):
            task.compute_error(self.configuration)

    def test_target_dim_mismatch(self):
        """Exception raised when the target dimension is unexpected."""
        task = JointVelocityTask(cost=1.0)
        task.set_target(np.zeros(self.nv), self.dt)
        with self.assertRaises(TaskDefinitionError):
            task.compute_error(self.configuration)

    def test_matrix_shapes(self):
        """Check the shapes of the error vector and Jacobian matrix."""
        task = JointVelocityTask(cost=1.0)
        task.set_target(np.zeros(self.nv - 6), self.dt)
        e = task.compute_error(self.configuration)
        J = task.compute_jacobian(self.configuration)
        self.assertEqual(e.shape[0], J.shape[0])

    def test_dt(self):
        """Check the shapes of the error vector and Jacobian matrix."""
        task = JointVelocityTask(cost=1.0)

        dt_1 = 3e-3  # seconds
        task.set_target(np.ones(self.nv - 6), dt_1)
        e_1 = task.compute_error(self.configuration)
        self.assertAlmostEqual(e_1[0], dt_1)  # unit target

        dt_2 = 7e-3  # seconds
        task.set_target(np.ones(self.nv - 6), dt_2)
        e_2 = task.compute_error(self.configuration)
        self.assertAlmostEqual(e_2[0], dt_2)  # unit target
