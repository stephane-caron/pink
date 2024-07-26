#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

"""Test fixture for the low-acceleration task."""

import unittest

import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.tasks import LowAccelerationTask


class TestLowAccelerationTask(unittest.TestCase):
    """Test consistency of the low-acceleration task."""

    def setUp(self):
        """Prepare test fixture."""
        robot = load_robot_description("ur3_description", root_joint=None)
        self.configuration = Configuration(robot.model, robot.data, robot.q0)

    def test_task_repr(self):
        """String representation reports the task gain, costs and target."""
        task = LowAccelerationTask(cost=1.0)
        self.assertTrue("cost=" in repr(task))
        self.assertFalse("gain=" in repr(task))
        self.assertFalse("lm_damping=" in repr(task))

    def test_qp_objective(self):
        task = LowAccelerationTask(cost=1.0)
        nv = self.configuration.model.nv
        v_prev = np.array([1.0, 2.0, 3.0, 4.0, -3.0, -2.0])
        dt = 1.234e-2  # [s]
        task.set_last_integration(v_prev, dt)
        H, c = task.compute_qp_objective(self.configuration)
        self.assertLess(np.linalg.norm(H - np.eye(nv)), 1e-10)
        self.assertLess(np.linalg.norm(c + v_prev * dt), 1e-10)
