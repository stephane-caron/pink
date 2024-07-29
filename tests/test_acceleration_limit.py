#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

"""Test acceleration limit."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink.limits import AccelerationLimit


class TestAccelerationLimit(unittest.TestCase):
    """Test fixture for acceleration limit."""

    def setUp(self):
        """Set test fixture up."""
        robot = load_robot_description("ur3_description")
        model = robot.model
        a_max = np.ones(robot.model.nv)
        self.a_max = a_max
        self.limit = AccelerationLimit(model, a_max)
        self.model = model

    def test_dimensions(self):
        """Check dimensions of configuration limit projection."""
        nb = len(self.limit.indices)
        nv = self.model.nv
        self.assertEqual(self.limit.projection_matrix.shape, (nb, nv))

    def test_model_with_no_limit(self):
        """Check that unbounded models don't fail."""
        empty_model = pin.Model()
        empty_bounded = AccelerationLimit(empty_model, self.a_max)
        self.assertEqual(len(empty_bounded.indices), 0)
        self.assertIsNone(empty_bounded.compute_qp_inequalities(np.empty(0), 1e-3))
