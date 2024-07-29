#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

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
        """Check dimensions of velocity limit projection."""
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
        empty_bounded = VelocityLimit(empty_model)
        self.assertEqual(len(empty_bounded.indices), 0)
