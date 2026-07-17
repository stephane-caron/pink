#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Test velocity limit."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink.exceptions import PinkError
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

    def test_velocity_limit_argument_overrides_model(self):
        """A given velocity_limit takes precedence over the model's."""
        nv = self.model.nv
        limit = VelocityLimit(self.model, velocity_limit=2.0 * np.ones(nv))
        # Every tangent coordinate now carries a finite limit, so every
        # joint is bounded regardless of what the model declares.
        self.assertEqual(len(limit.indices), nv)
        # The QP bound uses the argument, not model.velocityLimit.
        _, h = limit.compute_qp_inequalities(configuration=None, dt=1e-3)
        self.assertTrue(np.allclose(h, 1e-3 * 2.0))

    def test_velocity_limit_argument_wrong_shape(self):
        """A velocity_limit of the wrong dimension is rejected."""
        with self.assertRaises(PinkError):
            VelocityLimit(
                self.model, velocity_limit=np.ones(self.model.nv + 1)
            )
