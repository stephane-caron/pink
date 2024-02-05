#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2024 Inria

"""Test task Jacobian matrices against finite differences."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.tasks import FrameTask, JointCouplingTask, PostureTask, Task


class TestJacobians(unittest.TestCase):
    """Test task Jacobian matrices against finite differences."""

    def setUp(self, nb_configs=10, nb_dirs=2):
        """Prepare test fixture."""
        np.random.seed(42)
        random_dq = 2.0 * np.random.random((nb_dirs, 6)) - 1.0
        l2norms = np.sqrt((random_dq * random_dq).sum(axis=1))
        random_dirs = random_dq / l2norms.reshape((nb_dirs, 1))
        random_q = (
            0.1 * np.pi * (2.0 * np.random.random((nb_configs, 6)) - 1.0)
        )
        robot = load_robot_description("ur3_description")
        self.assertEqual(robot.nq, 6)
        self.link = "ee_link"
        self.data = robot.data
        self.model = robot.model
        self.random_dirs = random_dirs
        self.random_q = random_q
        self.robot = robot

    def check_jacobian_finite_diff(self, task: Task, tol: float):
        """Check that a task Jacobian is de/dq by finite differences.

        Args:
            task: Task to test the Jacobian of.
            tol: Test tolerance.
        """

        def e(q):
            configuration = Configuration(self.model, self.data, q)
            return task.compute_error(configuration)

        def J(q):
            configuration = Configuration(self.model, self.data, q)
            return task.compute_jacobian(configuration)

        nq = self.model.nq
        nv = self.model.nv
        for q_0 in self.random_q:
            J_0 = J(q_0)
            e_0 = e(q_0)

            J_finite = np.empty((e_0.shape[0], nv))
            for i in range(nq):
                h = 0.000001
                e_i = np.eye(nq)[i]
                J_finite[:, i] = (e(q_0 + h * e_i) - e_0) / h

            self.assertLess(np.linalg.norm(J_0 - J_finite, ord=np.inf), tol)

    def test_frame_task(self):
        frame_task = FrameTask(
            self.link,
            position_cost=1.0,
            orientation_cost=1.0,
        )
        frame_task.set_target(pin.SE3.Random())
        self.check_jacobian_finite_diff(frame_task, tol=1e-5)

    def test_joint_coupling_task(self):
        configuration = Configuration(self.model, self.data, self.robot.q0)
        joint_coupling_task = JointCouplingTask(
            ["shoulder_lift_joint", "shoulder_pan_joint"],
            [1.0, -1.0],
            100.0,
            configuration,
        )
        self.check_jacobian_finite_diff(joint_coupling_task, tol=1e-6)

    def test_posture_task(self):
        posture_task = PostureTask(cost=1.0)
        posture_task.set_target(self.robot.q0)
        self.check_jacobian_finite_diff(posture_task, tol=1e-6)
