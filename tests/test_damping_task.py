#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

"""Test fixture for the damping task."""

import unittest

import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.tasks import DampingTask


class TestDampingTask(unittest.TestCase):
    """Test consistency of the damping task."""

    def setUp(self):
        """Prepare test fixture."""
        robot = load_robot_description("ur3_description", root_joint=None)
        self.configuration = Configuration(robot.model, robot.data, robot.q0)

    def test_task_repr(self):
        """String representation reports the task gain, costs and target."""
        task = DampingTask(cost=1.0)
        self.assertTrue("cost=" in repr(task))
        self.assertFalse("gain=" in repr(task))
        self.assertFalse("lm_damping=" in repr(task))

    def test_qp_objective(self):
        task = DampingTask(cost=1.0)
        nv = self.configuration.model.nv
        H, c = task.compute_qp_objective(self.configuration)
        self.assertLess(np.linalg.norm(H - np.eye(nv)), 1e-10)
        self.assertLess(np.linalg.norm(c), 1e-10)
