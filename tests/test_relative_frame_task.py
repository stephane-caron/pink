#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

"""Test fixture for the relative frame task."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.tasks import RelativeFrameTask


class TestRelativeFrameTask(unittest.TestCase):
    """Test consistency of the relative frame task."""

    def setUp(self):
        """Prepare test fixture."""
        robot = load_robot_description(
            "stretch_description", root_joint=pin.JointModelPlanar()
        )
        self.configuration = Configuration(robot.model, robot.data, robot.q0)

    def test_set_target_from_configuration(self):
        """Read target after setting it to a known value."""
        task = RelativeFrameTask(
            "link_gripper_fingertip_right",
            "base_link",
            position_cost=1.0,
            orientation_cost=0.1,
        )
        task.set_target_from_configuration(self.configuration)
        transform_ankle_to_world = self.configuration.get_transform(
            "l_ankle", "base_link"
        )
        self.assertTrue(
            np.allclose(
                transform_ankle_to_world, task.transform_target_to_world
            )
        )

    def test_task_repr(self):
        """String representation reports the task gain, costs and target."""
        earflap_task = RelativeFrameTask(
            "earflap",
            "groot",
            position_cost=1.0, orientation_cost=0.1
        )
        self.assertTrue("frame=" in repr(earflap_task))
        self.assertTrue("root=" in repr(earflap_task))
        self.assertTrue("gain=" in repr(earflap_task))
        self.assertTrue("cost=" in repr(earflap_task))
        self.assertTrue("transform_target_to_root=" in repr(earflap_task))

