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
Test task Jacobian matrices against finite differences.
"""

import unittest

import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description

import pink
from pink.tasks import BodyTask


class TestJacobians(unittest.TestCase):

    """
    Test task Jacobian matrices against finite differences.
    """

    def setUp(self):
        self.robot = load_robot_description("ur3_description")
        self.body = "ee_link"
        self.assertEqual(self.robot.nq, 6)

    def test_body_task(self):
        """
        Test BodyTask Jacobian matrix.
        """
        task = BodyTask(self.body, position_cost=1.0, orientation_cost=1.0)
        q_0 = pink.custom_configuration_vector(
            self.robot,
            shoulder_lift_joint=1.0,
            shoulder_pan_joint=1.0,
            elbow_joint=1.0,
        )
        configuration = pink.apply_configuration(self.robot, q_0)
        task.set_target_from_configuration(configuration)

        def e(q):
            configuration = pink.apply_configuration(self.robot, q)
            return task.compute_error_in_body(configuration)

        def J(q):
            configuration = pink.apply_configuration(self.robot, q)
            jacobian_in_body = configuration.get_body_jacobian(self.body)
            return jacobian_in_body

        # e(q_0) == 0
        self.assertLess(np.linalg.norm(e(q_0), ord=np.inf), 1e-20)

        nq = self.robot.model.nq
        basis = np.eye(nq)
        for i in range(nq):
            h = 1e-3
            finite_diff = e(q_0 + h * basis[i]) / h
            linear_approx = J(q_0) @ basis[i]

            tol = 1e-6
            self.assertLess(
                np.linalg.norm(finite_diff - linear_approx, ord=np.inf), tol
            )
