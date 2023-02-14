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

"""Test configuration limit."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.limits import ConfigurationLimit


class TestConfigurationLimit(unittest.TestCase):
    """Test fixture for configuration limit."""

    def setUp(self):
        """Set test fixture up."""
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        model = robot.model
        self.data = robot.data
        self.limit = ConfigurationLimit(model)
        self.model = model
        self.robot = robot

    def test_dimensions(self):
        """Check dimensions of configuration limit projection."""
        for joint in self.limit.joints:
            self.assertGreaterEqual(joint.idx_q, 0)
            self.assertGreaterEqual(joint.idx_v, 0)
        nb = len(self.limit.joints)  # those are only revolute joints
        nv = self.model.nv
        self.assertEqual(len(self.limit.indices), nb)
        self.assertEqual(self.limit.projection_matrix.shape, (nb, nv))

    def test_model_with_no_limit(self):
        """Check that unbounded models don't fail."""
        empty_model = pin.Model()
        empty_bounded = ConfigurationLimit(empty_model)
        self.assertEqual(len(empty_bounded.indices), 0)

    def test_far_from_limit(self, tol=1e-10):
        """Limit has no effect when the configuration is far away.

        Args:
            tol: Numerical tolerance.

        When we are far away from configuration limits, the velocity limit is
        simply the configuration-agnostic one from the robot.

        Note:
            This test works with a commit-pinned description of Upkie. It will
            not work for any description, as a robot may have e.g. some
            velocity-unbounded joints.
        """
        robot = load_robot_description(
            "upkie_description",
            root_joint=pin.JointModelFreeFlyer(),
            # https://github.com/robot-descriptions/robot_descriptions.py/issues/31
            # commit="62f3ba24c2045b44faedb7c6c6167e74e157b49e",
        )
        configuration = Configuration(robot.model, robot.data, robot.q0)
        limit = ConfigurationLimit(robot.model)
        G, h = limit.compute_qp_inequalities(robot.q0, dt=1e-3)
        v_lim = configuration.model.velocityLimit
        self.assertLess(np.max(+G @ v_lim - h), -tol)
        self.assertLess(np.max(-G @ v_lim - h), -tol)

    def test_configuration_limit_repulsion(self, tol=1e-10):
        """Velocities are scaled down when close to a configuration limit.

        Args:
            tol: Numerical tolerance.
        """
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        dt = 1e-3  # [s]
        configuration = Configuration(robot.model, robot.data, robot.q0)
        slack_vel = 5.5e-4  # [rad] / [s]

        # Clamp configuration limit to q +/- slack_vel * dt
        robot.model.lowerPositionLimit = configuration.integrate(
            -slack_vel * configuration.tangent.ones, dt
        )
        robot.model.upperPositionLimit = configuration.integrate(
            +slack_vel * configuration.tangent.ones, dt
        )

        limit = ConfigurationLimit(robot.model, config_limit_gain=0.5)
        G, h = limit.compute_qp_inequalities(configuration.q, dt)
        self.assertLess(np.max(h), slack_vel * dt + tol)
        self.assertGreater(np.min(h), -slack_vel * dt - tol)
