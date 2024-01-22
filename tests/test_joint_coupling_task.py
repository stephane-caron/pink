#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Test fixture for the joint coupling task."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.tasks import JointCouplingTask


class TestJointCouplingTask(unittest.TestCase):
    """Test consistency of the joint coupling task.

    Note:
        This fixture only tests the task itself. Integration tests with the IK
        are carried out in :class:`TestSolveIK`.
    """

    def setUp(self):
        """Prepare test fixture."""
        robot = load_robot_description(
            "draco3_description", root_joint=pin.JointModelFreeFlyer()
        )
        self.configuration = Configuration(robot.model, robot.data, robot.q0)

    def test_task_repr(self):
        """String representation reports the task gain, costs and target."""
        task = JointCouplingTask(
            joint_names=["r_knee_fe_jp", "r_knee_fe_jd"],
            ratios=[1.0, -1.0],
            cost=1.0,
            configuration=self.configuration,
        )
        self.assertTrue("cost=" in repr(task))
        self.assertTrue("gain=" in repr(task))

    def test_unit_cost_qp_objective(self):
        """A unit cost vector means the QP objective is (J^T J, -e^T J)."""
        task = JointCouplingTask(
            joint_names=["r_knee_fe_jp", "r_knee_fe_jd"],
            ratios=[1.0, -1.0],
            cost=1.0,
            configuration=self.configuration,
        )
        e = task.compute_error(self.configuration)
        J = task.compute_jacobian(self.configuration)
        H, c = task.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(J.T @ J, H))
        self.assertTrue(np.allclose(-e.T @ J, c))

    def test_zero_cost_same_as_disabling_task(self):
        """The task has no effect when its cost is zero."""
        task = JointCouplingTask(
            joint_names=["r_knee_fe_jp", "r_knee_fe_jd"],
            ratios=[1.0, -1.0],
            cost=0.0,
            configuration=self.configuration,
        )
        J = task.compute_jacobian(self.configuration)
        H, c = task.compute_qp_objective(self.configuration)
        qd = np.random.random(J.shape[1:])
        cost = qd.T @ H @ qd + c @ qd
        self.assertAlmostEqual(cost, 0.0)
