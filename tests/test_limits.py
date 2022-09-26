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
        self.dt = 1e-3  # [s]

    def test_forward_velocity_limit(self):
        """
        When we are far away from configuration limits, the velocity limit is
        simply the configuration-agnostic one from the robot.
        """
        configuration = apply_configuration(self.robot, self.robot.q0)
        v_max, v_min = compute_velocity_limits(configuration, self.dt)
        model_v_lim = configuration.model.velocityLimit
        self.assertTrue(np.allclose(v_max, +model_v_lim))
        self.assertTrue(np.allclose(v_min, -model_v_lim))

    def test_configuration_limit_repulsion(self):
        """
        Velocities are scaled down when close to a configuration limit.
        """
        configuration = apply_configuration(self.robot, self.robot.q0)
        slack_vel = 5.5e-4 * configuration.tangent.ones
        self.robot.model.upperPositionLimit = configuration.integrate(
            slack_vel, self.dt
        )
        v_max, v_min = compute_velocity_limits(
            configuration, self.dt, config_limit_gain=0.5
        )
        self.assertTrue(np.all(v_max < self.robot.model.velocityLimit))
        self.assertTrue(np.all(v_max < slack_vel))
