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

"""Test the Configuration type."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.exceptions import BodyNotFound, NotWithinConfigurationLimits


class TestConfiguration(unittest.TestCase):
    def test_constructor(self):
        """Constructing a configuration computes Jacobians."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        robot.data.J.fill(42.0)
        configuration = Configuration(robot.model, robot.data, robot.q0)
        J_check = np.array(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.096,
                    0.389,
                    0.0,
                    0.746,
                    -0.0,
                    -0.0,
                    -0.096,
                    0.389,
                    0.0,
                    0.746,
                    0.0,
                    -0.192,
                    0.0,
                    -0.522,
                    0.0,
                    0.24,
                    -0.217,
                    0.24,
                    0.0,
                    0.24,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.645,
                    -0.522,
                    -0.0,
                    -0.24,
                    -0.217,
                    -0.24,
                    0.0,
                    -0.24,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.746,
                    -0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.746,
                    -0.0,
                    0.0,
                    0.0,
                    0.192,
                    0.0,
                    0.522,
                    0.0,
                    0.0,
                    -0.0,
                    -0.022,
                    -0.0,
                    -0.124,
                    -0.169,
                    -0.124,
                    -0.169,
                    -0.106,
                    -0.151,
                    0.003,
                    0.645,
                    0.0,
                    0.0,
                    0.522,
                    0.0,
                    0.0,
                    -0.0,
                    -0.022,
                    -0.0,
                    -0.124,
                    -0.169,
                    -0.124,
                    -0.169,
                    -0.106,
                    -0.151,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.096,
                    0.0,
                    -0.02,
                    -0.096,
                    0.02,
                    0.0,
                    0.096,
                    0.0,
                    -0.02,
                    0.096,
                    0.02,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.24,
                    0.0,
                    0.004,
                    0.0,
                    -0.24,
                    0.0,
                    -0.246,
                    -0.246,
                    -0.246,
                    -0.246,
                    -0.217,
                    -0.217,
                    -0.0,
                    -0.0,
                    -0.003,
                    0.0,
                    0.24,
                    0.0,
                    0.004,
                    0.0,
                    0.24,
                    0.0,
                    0.246,
                    0.246,
                    0.246,
                    0.246,
                    0.217,
                    0.217,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        )
        self.assertTrue(np.allclose(configuration.data.J, J_check))

    def test_copy_no_forward_kinematics(self):
        """Refer to the input data, compute forward kinematics."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        robot.data.J.fill(42.0)
        configuration = Configuration(
            robot.model,
            robot.data,
            robot.q0,
            copy_data=False,
            forward_kinematics=True,
        )
        self.assertFalse(np.allclose(configuration.data.J, 42.0))
        configuration.data.J.fill(12.0)
        self.assertTrue(np.allclose(robot.data.J, 12.0))

    def test_no_copy_no_forward_kinematics(self):
        """Not copying means working directly on the underlying data."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        robot.data.J.fill(42.0)
        configuration = Configuration(
            robot.model,
            robot.data,
            robot.q0,
            copy_data=False,
            forward_kinematics=False,
        )
        self.assertTrue(np.allclose(configuration.data.J, 42.0))
        configuration.data.J.fill(12.0)
        self.assertTrue(np.allclose(robot.data.J, 12.0))

    def test_transform_found(self):
        """Return the pose of an existing robot body."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        transform_pelvis_to_world = configuration.get_transform_body_to_world(
            "PELVIS_S"
        )
        self.assertTrue(
            np.allclose(
                transform_pelvis_to_world.np[3, :],
                np.array([0.0, 0.0, 0.0, 1.0]),
            )
        )

    def test_transform_not_found(self):
        """Raise an error when the request robot body is not found."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        with self.assertRaises(KeyError):
            configuration.get_transform_body_to_world("foo")

    def test_check_limits(self):
        """Raise an error if and only if a joint limit is exceened."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        q = robot.q0
        configuration = Configuration(robot.model, robot.data, q)
        configuration.check_limits()
        q[-10] += 1e4  # move configuration out of bounds
        configuration = Configuration(robot.model, robot.data, q)
        with self.assertRaises(NotWithinConfigurationLimits):
            configuration.check_limits()

    def test_q_is_a_read_only_copy(self):
        """The `q` attribute of a configuration is a read-only copy."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        original_q = robot.q0
        configuration = Configuration(robot.model, robot.data, original_q)
        the_answer = 42.0
        self.assertNotEqual(configuration.q[2], the_answer)
        original_q[2] = the_answer
        self.assertNotEqual(configuration.q[2], the_answer)
        with self.assertRaises(ValueError):
            configuration.q[2] += 3.0  # read-only

    def test_tangent_eye(self):
        """Configuration's tangent eye is an identity matrix."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        v = np.array([i for i in range(robot.model.nv)])
        self.assertTrue(np.allclose(configuration.tangent.eye.dot(v), v))

    def test_tangent_ones(self):
        """Configuration's tangent ones is a vector of 1.0's."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        self.assertEqual(np.sum(configuration.tangent.ones), robot.model.nv)

    def test_tangent_zeros(self):
        """Configuration's tangent ones is a vector of 0.0's."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        self.assertAlmostEqual(np.sum(configuration.tangent.zeros), 0.0)
        self.assertEqual(len(configuration.tangent.zeros), robot.model.nv)

    def test_get_body_jacobian(self):
        """
        Querying a body that does not exist raises a ValueError.
        """
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        with self.assertRaises(BodyNotFound):
            configuration.get_body_jacobian("does_not_exist")
