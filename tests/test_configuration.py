#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Test the configuration type."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.exceptions import FrameNotFound, NotWithinConfigurationLimits


class TestConfiguration(unittest.TestCase):
    """Test configuration type."""

    def test_constructor(self):
        """Constructing a configuration computes Jacobians."""
        robot = load_robot_description(
            "jvrc_description",
            root_joint=pin.JointModelFreeFlyer(),
            commit="9ff8efbc7043459a8f0892662bd030d8020fb682",
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
                    -0.106,
                    -0.151,
                    -0.106,
                    -0.151,
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
                    -0.106,
                    -0.151,
                    -0.106,
                    -0.151,
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
                    -0.256,
                    -0.256,
                    -0.256,
                    -0.256,
                    -0.207,
                    -0.207,
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
                    0.256,
                    0.256,
                    0.256,
                    0.256,
                    0.207,
                    0.207,
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
        transform_pelvis_to_world = configuration.get_transform_frame_to_world(
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
        with self.assertRaises(FrameNotFound):
            configuration.get_transform_frame_to_world("foo")

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

    def test_frame_jacobian_not_found(self):
        """Querying a body that does not exist raises a ValueError."""
        robot = load_robot_description(
            "jvrc_description", root_joint=pin.JointModelFreeFlyer()
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        with self.assertRaises(FrameNotFound):
            configuration.get_frame_jacobian("does_not_exist")

    def test_get_integrate_inplace(self):
        """Test in-place integration."""
        robot = load_robot_description("sigmaban_description", root_joint=None)
        configuration = Configuration(robot.model, robot.data, robot.q0)
        velocity = robot.model.tangent.ones
        configuration.integrate_inplace(velocity, dt=1e-3)
        self.assertGreater(np.linalg.norm(configuration.q - robot.q0), 2e-3)
