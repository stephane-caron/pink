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

import os
import unittest

import numpy as np

from pink import apply_configuration
from pink.limits import compute_velocity_limits
from pink.models import build_from_urdf


class TestLimits(unittest.TestCase):
    def setUp(self):
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        upkie_description = os.path.join(models_dir, "upkie_description")
        robot = build_from_urdf(upkie_description)
        configuration = apply_configuration(robot, robot.q0)
        self.dt = 1e-3
        self.configuration = configuration
        self.robot = robot

    def test_forward_velocity_limit(self):
        """
        When we are far away from configuration limits, the velocity limit is
        simply the configuration-agnostic one from the robot.
        """
        v_max, v_min = compute_velocity_limits(self.configuration, self.dt)
        velocity_limit = self.configuration.model.velocityLimit
        self.assertTrue(np.allclose(v_max, +velocity_limit))
        self.assertTrue(np.allclose(v_min, -velocity_limit))

    def test_configuration_limit_repulsion(self):
        """
        Velocities are scaled down when close to a configuration limit.
        """
        q = self.configuration.q
        slack_vel = 5.5e-4  # [rad] / [s]
        self.robot.model.upperPositionLimit = q + slack_vel * self.dt
        v_max, v_min = compute_velocity_limits(
            self.configuration, self.dt, config_limit_gain=0.5
        )
        default_limit = self.robot.model.velocityLimit
        self.assertTrue(np.allclose(v_max[:6], default_limit[:6]))
        self.assertTrue(np.all(v_max[6:] < default_limit[6:]))
        self.assertTrue(np.all(v_max[6:] < slack_vel))
