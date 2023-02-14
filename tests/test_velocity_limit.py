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

"""Test velocity limit."""

import unittest

import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink.limits import VelocityLimit


class TestVelocityLimit(unittest.TestCase):

    """Test fixture for velocity limit."""

    def setUp(self):
        """Set test fixture up."""
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        model = robot.model
        self.limit = VelocityLimit(model)
        self.model = model

    def test_dimensions(self):
        """Check dimensions of configuration limit projection."""
        for joint in self.limit.joints:
            self.assertGreaterEqual(joint.idx_q, 0)
            self.assertGreaterEqual(joint.idx_v, 0)
        nb = len(self.limit.joints)  # those are only revolute joints
        nv = self.model.nv
        self.assertEqual(self.limit.dim, nb)
        self.assertEqual(self.limit.projection_matrix.shape, (nb, nv))

    def test_model_with_no_limit(self):
        """Check that unbounded models don't fail."""
        empty_model = pin.Model()
        empty_bounded = VelocityLimit(empty_model)
        self.assertEqual(empty_bounded.dim, 0)
