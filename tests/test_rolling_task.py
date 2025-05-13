#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Inria

"""Test fixture for the rolling task."""

import unittest

import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.tasks import RollingTask


class TestRollingTask(unittest.TestCase):
    """Test consistency of the rolling task.

    Note:
        This fixture only tests the task itself. Integration tests with the IK
        are carried out in :class:`TestSolveIK`.
    """

    def setUp(self):
        robot = load_robot_description("upkie_description")
        left_wheel_task = RollingTask(
            "left_wheel_center",
            floor_frame="universe",
            wheel_radius=0.1,
            cost=1.0,
        )
        self.configuration = Configuration(robot.model, robot.data, robot.q0)
        self.left_wheel_task = left_wheel_task

    def test_compute_error(self):
        task = self.left_wheel_task
        error_in_rim = task.compute_error(self.configuration)
        self.assertAlmostEqual(np.linalg.norm(error_in_rim[:2]), 0.0)

    def test_compute_jacobian(self):
        task = self.left_wheel_task
        jacobian_hub_in_rim = task.compute_jacobian(self.configuration)
        self.assertEqual(
            jacobian_hub_in_rim.shape, (3, self.configuration.model.nv)
        )
