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

"""Test task Jacobian matrices against finite differences."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

import pink
from pink.tasks import BodyTask


class TestJacobians(unittest.TestCase):

    """
    Test task Jacobian matrices against finite differences.
    """

    def setUp(self, nb_configs=10, nb_dirs=2):
        np.random.seed(42)
        random_dq = 2.0 * np.random.random((nb_dirs, 6)) - 1.0
        l2norms = np.sqrt((random_dq * random_dq).sum(axis=1))
        random_dirs = random_dq / l2norms.reshape((nb_dirs, 1))
        random_q = (
            0.1 * np.pi * (2.0 * np.random.random((nb_configs, 6)) - 1.0)
        )
        robot = load_robot_description("ur3_description")
        self.assertEqual(robot.nq, 6)
        self.body = "ee_link"
        self.random_dirs = random_dirs
        self.random_q = random_q
        self.robot = robot

    def test_body_task(self):
        """
        Test BodyTask Jacobian matrix.
        """
        task = BodyTask(self.body, position_cost=1.0, orientation_cost=1.0)
        task.set_target(pin.SE3.Random())

        def e(q):
            configuration = pink.apply_configuration(self.robot, q)
            return task.compute_error_in_body(configuration)

        def J(q):
            configuration = pink.apply_configuration(self.robot, q)
            jacobian, _ = task.compute_task_dynamics(configuration, version=3)
            return jacobian

        nq = self.robot.model.nq
        nv = self.robot.model.nv
        for q_0 in self.random_q:
            J_0 = J(q_0)
            e_0 = e(q_0)

            J_finite = np.empty((e_0.shape[0], nv))
            for i in range(nq):
                h = 0.000001
                e_i = np.eye(nq)[i]
                J_finite[:, i] = (e(q_0 + h * e_i) - e_0) / h

            print(np.linalg.norm(J_0 + J_finite, ord=np.inf))
