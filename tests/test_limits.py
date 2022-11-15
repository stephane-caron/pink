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

from pink import apply_configuration
from pink.limits import compute_velocity_limits


class TestLimits(unittest.TestCase):
    def setUp(self):
        self.robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )

    def test_limit_dimension(self):
        """
        Check that velocity limit vectors have the same dimension as the number
        of bounded joints.
        """
        dt = 1e-3  # [s]
        model = self.robot.model
        configuration = apply_configuration(self.robot, self.robot.q0)
        v_max, v_min = compute_velocity_limits(configuration, dt)
        self.assertEqual(v_max.shape, (model.nv_bounded,))
        self.assertEqual(v_min.shape, (model.nv_bounded,))

    def test_forward_velocity_limit(self):
        """
        When we are far away from configuration limits, the velocity limit is
        simply the configuration-agnostic one from the robot.
        """
        dt = 1e-3  # [s]
        configuration = apply_configuration(self.robot, self.robot.q0)
        v_max, v_min = compute_velocity_limits(configuration, dt)
        v_lim = configuration.model.bounded.velocity_limit
        tol = 1e-10
        self.assertLess(np.max(v_max - v_lim), tol)
        self.assertLess(np.max(-v_lim - v_min), tol)

    def test_configuration_limit_repulsion(self):
        """
        Velocities are scaled down when close to a configuration limit.
        """
        dt = 1e-3  # [s]
        model = self.robot.model
        configuration = apply_configuration(self.robot, self.robot.q0)
        slack_vel = 5.5e-4 * configuration.tangent.ones
        bounded_slack_vel = slack_vel[model.bounded_tangent_idx]
        model.upperPositionLimit = configuration.integrate(slack_vel, dt)
        v_max, v_min = compute_velocity_limits(
            configuration, dt, config_limit_gain=0.5
        )
        tol = 1e-10
        self.assertLess(np.max(v_max - model.bounded_velocity_limit), tol)
        self.assertLess(np.max(v_max - bounded_slack_vel), tol)
