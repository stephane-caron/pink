#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Domrachev Ivan, Simeon Nedelchev

"""Test fixture for the com task."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.exceptions import TargetNotSet
from pink.tasks import ComTask


class TestComTask(unittest.TestCase):
    """Test consistency of the com task.

    Note:
        This fixture only tests the task itself. Integration tests with the IK
        are carried out in :class:`TestSolveIK`.
    """

    def setUp(self):
        """Prepare test fixture."""
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        self.configuration = Configuration(robot.model, robot.data, robot.q0)

    def test_task_repr(self):
        """String representation reports the task gain, costs and target."""
        task = ComTask(cost=1.0)
        self.assertTrue("cost=" in repr(task))
        self.assertTrue("gain=" in repr(task))
        self.assertTrue("target_com=" in repr(task))

    def test_target_not_set(self):
        """Raise an exception when the target is not set."""
        task = ComTask(cost=1.0)
        with self.assertRaises(TargetNotSet):
            task.compute_error(self.configuration)

    def test_set_target_from_configuration(self):
        """Check that target is set from its configuration value."""
        task = ComTask(cost=1.0)
        task.set_target_from_configuration(self.configuration)
        com = pin.centerOfMass(
            self.configuration.model,
            self.configuration.data,
            self.configuration.q,
        )
        self.assertTrue(np.allclose(com, task.target_com))

    def test_target_set_properly(self):
        """Return target properly once it's set."""
        task = ComTask(cost=1.0)
        com_task = np.random.rand(3)
        task.set_target(com_task)
        self.assertIsNotNone(task.target_com)
        if task.target_com is not None:  # help mypy
            self.assertTrue(np.allclose(com_task, task.target_com))

    def test_target_is_a_copy(self):
        """Target is saved as a copy, not a reference to the original."""
        task = ComTask(cost=1.0)
        com_des = np.random.rand(3)
        task.set_target(com_des)
        self.assertIsNotNone(task.target_com)

    def test_zero_error_when_target_at_body(self):
        """Error is zero when the target and body are at the same location."""
        task = ComTask(cost=1.0)
        task.set_target(
            pin.centerOfMass(self.configuration.model, self.configuration.data)
        )  # error == 0
        e = task.compute_error(self.configuration)
        self.assertLess(np.linalg.norm(e), 1e-10)

    def test_unit_cost_qp_objective(self):
        """A unit cost vector means the QP objective is (J^T J, -e^T J)."""
        task = ComTask(cost=1.0)
        task.set_target(
            pin.centerOfMass(self.configuration.model, self.configuration.data)
        )
        q_new = self.configuration.q.copy()
        q_new[1] += 1.0
        q_new[3] += 1.0
        q_new[5] += 1.0
        self.configuration.q = q_new
        e = task.compute_error(self.configuration)
        J = task.compute_jacobian(self.configuration)
        H, c = task.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(J.T @ J, H))
        print
        self.assertTrue(np.allclose(e.T @ J, c))

    def test_zero_cost_same_as_disabling_task(self):
        """The task has no effect when its cost is zero."""
        task = ComTask(cost=0.0)
        task.set_target(
            pin.centerOfMass(self.configuration.model, self.configuration.data)
        )
        J = task.compute_jacobian(self.configuration)
        H, c = task.compute_qp_objective(self.configuration)
        qd = np.random.random(J.shape[1:])
        cost = qd.T @ H @ qd + c @ qd
        self.assertAlmostEqual(cost, 0.0)

    def test_cost_raises_exception(self):
        """The task raises an exception when its cost is
        negative and allows float, list and np.ndarray as input."""
        with self.assertRaises(AssertionError):
            task = ComTask(cost=-1.0)
        with self.assertRaises(AssertionError):
            task = ComTask(cost=[-1.0, -1.0, -1.0])
        with self.assertRaises(AssertionError):
            task = ComTask(cost=-np.ones(3))

        task = ComTask(cost=1.0)
        task.set_cost(cost=[1.0, 1.0, 1.0])
        task.set_cost(cost=np.ones(3))

    def test_jac_without_target(self):
        """The jacobian call should raise an exception
        when the target is not set"""

        task = ComTask(cost=1.0)
        with self.assertRaises(TargetNotSet):
            task.compute_jacobian(self.configuration)
