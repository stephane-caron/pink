#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Test fixture for the linear holonomic task."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.exceptions import TaskDefinitionError, TaskJacobianNotSet
from pink.tasks import LinearHolonomicTask
from pink.utils import get_joint_idx


class TestLinearHolonomicTask(unittest.TestCase):
    """Test consistency of the linear holonomic task.

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

    def test_inconsistent_definition(self):
        """Exception when the task is not defined properly."""
        with self.assertRaises(TaskDefinitionError):
            LinearHolonomicTask(A=np.ones((3, 4)), b=np.ones(5), q_0=None)

    def test_task_repr(self):
        """String representation reports the task gain, costs and target."""
        A = np.zeros((1, self.configuration.model.nv))
        _, r_knee_fe_jp_v_idx = get_joint_idx(
            self.configuration.model, "r_knee_fe_jp"
        )
        _, r_knee_fe_jd_v_idx = get_joint_idx(
            self.configuration.model, "r_knee_fe_jd"
        )
        A[:, r_knee_fe_jp_v_idx] = 1.0
        A[:, r_knee_fe_jd_v_idx] = -1.0
        task = LinearHolonomicTask(A=A, b=np.zeros(1), q_0=None, cost=1.0)
        self.assertTrue("cost=" in repr(task))
        self.assertTrue("gain=" in repr(task))

    def test_jacobian_not_set_properly(self):
        """Raise an exception when the task Jacobian is not set properly."""
        A = np.zeros((1, self.configuration.model.nq))
        task = LinearHolonomicTask(A=A, b=np.zeros(1), q_0=None, cost=1.0)
        with self.assertRaises(TaskJacobianNotSet):
            task.compute_error(self.configuration)
        with self.assertRaises(TaskJacobianNotSet):
            task.compute_jacobian(self.configuration)

    def test_unit_cost_qp_objective(self):
        """A unit cost vector means the QP objective is (J^T J, -e^T J)."""
        A = np.zeros((2, self.configuration.model.nv))
        _, r_knee_fe_jp_v_idx = get_joint_idx(
            self.configuration.model, "r_knee_fe_jp"
        )
        _, r_knee_fe_jd_v_idx = get_joint_idx(
            self.configuration.model, "r_knee_fe_jd"
        )
        A[:, r_knee_fe_jp_v_idx] = 1.0
        A[:, r_knee_fe_jd_v_idx] = -1.0
        _, l_knee_fe_jp_v_idx = get_joint_idx(
            self.configuration.model, "l_knee_fe_jp"
        )
        _, l_knee_fe_jd_v_idx = get_joint_idx(
            self.configuration.model, "l_knee_fe_jd"
        )
        A[:, l_knee_fe_jp_v_idx] = 1.0
        A[:, l_knee_fe_jd_v_idx] = -1.0
        task = LinearHolonomicTask(
            A=A, b=np.zeros(2), q_0=None, cost=[1.0, 1.0]
        )
        e = task.compute_error(self.configuration)
        J = task.compute_jacobian(self.configuration)
        H, c = task.compute_qp_objective(self.configuration)
        self.assertTrue(np.allclose(J.T @ J, H))
        self.assertTrue(np.allclose(-e.T @ J, c))

    def test_zero_cost_same_as_disabling_task(self):
        """The task has no effect when its cost is zero."""
        A = np.zeros((2, self.configuration.model.nv))
        _, r_knee_fe_jp_v_idx = get_joint_idx(
            self.configuration.model, "r_knee_fe_jp"
        )
        _, r_knee_fe_jd_v_idx = get_joint_idx(
            self.configuration.model, "r_knee_fe_jd"
        )
        A[:, r_knee_fe_jp_v_idx] = 1.0
        A[:, r_knee_fe_jd_v_idx] = -1.0
        _, l_knee_fe_jp_v_idx = get_joint_idx(
            self.configuration.model, "l_knee_fe_jp"
        )
        _, l_knee_fe_jd_v_idx = get_joint_idx(
            self.configuration.model, "l_knee_fe_jd"
        )
        A[:, l_knee_fe_jp_v_idx] = 1.0
        A[:, l_knee_fe_jd_v_idx] = -1.0
        task = LinearHolonomicTask(
            A=A, b=np.zeros(2), q_0=None, cost=[0.0, 0.0]
        )
        J = task.compute_jacobian(self.configuration)
        H, c = task.compute_qp_objective(self.configuration)
        qd = np.random.random(J.shape[1:])
        cost = qd.T @ H @ qd + c @ qd
        self.assertAlmostEqual(cost, 0.0)
