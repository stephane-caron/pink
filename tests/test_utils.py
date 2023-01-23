#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Stéphane Caron
# Copyright 2023 Inria
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

"""Test fixture for other library features."""

import unittest

import numpy as np
import pinocchio as pin
from pink.tasks.utils import body_minus, spatial_minus
from pink.utils import custom_configuration_vector
from robot_descriptions.loaders.pinocchio import load_robot_description


class TestUtils(unittest.TestCase):
    def test_custom_configuration_vector(self):
        """Check a custom configuration vector for Upkie.

        Assumes the left and right knees have joint indices respectively 8 and
        11 in the configuration vector.
        """
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        q = custom_configuration_vector(robot, left_knee=0.2, right_knee=-0.2)
        self.assertAlmostEqual(q[8], 0.2)
        self.assertAlmostEqual(q[11], -0.2)

    def test_minus(self):
        """Test Lie minus operators."""
        X = pin.SE3.Random()
        Y = pin.SE3.Random()
        self.assertTrue(np.allclose(Y, X * pin.exp6(body_minus(Y, X))))
        self.assertTrue(np.allclose(Y, pin.exp6(spatial_minus(Y, X)) * X))
