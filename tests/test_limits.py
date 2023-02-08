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
Test velocity and actuated-joint limits.
"""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.limits import compute_velocity_limits


class TestLimits(unittest.TestCase):
    def setUp(self):
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        self.data = robot.data
        self.model = robot.model
        self.robot = robot

    def test_limit_dimension(self):
        """Velocity limit dimension is the number of bounded joints."""
        configuration = Configuration(self.model, self.data, self.robot.q0)
        v_max, v_min = compute_velocity_limits(configuration, dt=1e-3)
        self.assertEqual(v_max.shape, (self.model.bounded_tangent.dim,))
        self.assertEqual(v_min.shape, (self.model.bounded_tangent.dim,))

    def test_model_with_no_joint_limit(self):
        """Model with no joint limit has no velocity-limit vector."""
        model = pin.Model()
        model.addJoint(
            0, pin.JointModelSpherical(), pin.SE3.Identity(), "spherical"
        )
        robot = pin.RobotWrapper(model=model)
        configuration = Configuration(robot.model, robot.data, robot.q0)
        v_max, v_min = compute_velocity_limits(configuration, dt=1e-3)
        self.assertIsNone(v_max)
        self.assertIsNone(v_min)

    def test_model_with_limitless_joint(self):
        """Same as previous test, but the joint has a limit set to zero."""
        model = pin.Model()
        model.addJoint(
            0,
            pin.JointModelRevoluteUnaligned(),
            pin.SE3.Identity(),
            "revolute",
            max_effort=np.array([0.0]),
            max_velocity=np.array([0.0]),
            min_config=np.array([0.0]),
            max_config=np.array([0.0]),
        )
        robot = pin.RobotWrapper(model=model)
        configuration = Configuration(robot.model, robot.data, robot.q0)
        v_max, v_min = compute_velocity_limits(configuration, dt=1e-3)
        self.assertIsNone(v_max)
        self.assertIsNone(v_min)

    def test_forward_velocity_limit(self):
        """Velocity limits have no effect far from configuration limits.

        When we are far away from configuration limits, the velocity limit is
        simply the configuration-agnostic one from the robot.
        """
        configuration = Configuration(self.model, self.data, self.robot.q0)
        v_max, v_min = compute_velocity_limits(configuration, dt=1e-3)
        v_lim = configuration.model.bounded_tangent.velocity_limit
        tol = 1e-10
        self.assertLess(np.max(v_max - v_lim), tol)
        self.assertLess(np.max(-v_lim - v_min), tol)

    def test_configuration_limit_repulsion(self):
        """Velocities are scaled down when close to a configuration limit."""
        dt = 1e-3  # [s]
        configuration = Configuration(self.model, self.data, self.robot.q0)
        slack_vel = 5.5e-4 * configuration.tangent.ones
        bounded_slack_vel = slack_vel[self.model.bounded_tangent.indices]
        self.model.upperPositionLimit = configuration.integrate(slack_vel, dt)
        v_max, v_min = compute_velocity_limits(
            configuration, dt, config_limit_gain=0.5
        )
        tol = 1e-10
        self.assertLess(
            np.max(v_max - self.model.bounded_tangent.velocity_limit), tol
        )
        self.assertLess(np.max(v_max - bounded_slack_vel), tol)

    def test_velocity_without_configuration_limits(self, tol: float = 1e-10):
        """Velocity limits are loaded for a model without config limits.

        Args:
            tol: Equality test tolerance.

        Note:
            The loaded robot description is commit-pinned to make sure we load
            a URDF without configuration limits.
        """
        dt = 1e-3  # [s]
        sigmaban = load_robot_description(
            "sigmaban_description",
            commit="d5d023fd35800d00d7647000bce8602617a4960d",
        )
        configuration = Configuration(sigmaban.model, sigmaban.data, sigmaban.q0)
        v_max, v_min = compute_velocity_limits(
            configuration, dt, config_limit_gain=0.5
        )
        self.assertIsNotNone(v_max)
        self.assertIsNotNone(v_min)
        self.assertLess(
            np.max(v_max - sigmaban.model.bounded_tangent.velocity_limit), tol
        )
