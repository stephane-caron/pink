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
from pink.tasks import FrameTask, RelativeFrameTask


class TestRelativeFrameTask(unittest.TestCase):
    """Test consistency of the relative frame task."""

    def setUp(self):
        """Prepare test fixture."""
        robot = load_robot_description(
            "stretch_description", root_joint=pin.JointModelPlanar()
        )
        self.configuration = Configuration(robot.model, robot.data, robot.q0)
        self.robot = robot

    def test_set_target_from_configuration(self):
        """Read target after setting it to a known value."""
        task = RelativeFrameTask(
            "link_gripper_fingertip_right",
            "base_link",
            position_cost=1.0,
            orientation_cost=0.1,
        )
        task.set_target_from_configuration(self.configuration)
        transform_fingertip_to_base = self.configuration.get_transform(
            "link_gripper_fingertip_right", "base_link"
        )
        self.assertTrue(
            np.allclose(
                transform_fingertip_to_base,
                task.transform_target_to_root,
            )
        )

    def test_task_repr(self):
        """String representation reports the task gain, costs and target."""
        earflap_task = RelativeFrameTask(
            "earflap", "groot", position_cost=1.0, orientation_cost=0.1
        )
        self.assertTrue("frame=" in repr(earflap_task))
        self.assertTrue("root=" in repr(earflap_task))
        self.assertTrue("gain=" in repr(earflap_task))
        self.assertTrue("cost=" in repr(earflap_task))
        self.assertTrue("transform_target_to_root=" in repr(earflap_task))

    def test_matches_frame_task(self):
        relative_task = RelativeFrameTask(
            "link_gripper_fingertip_right",
            "universe",
            position_cost=1.0,
            orientation_cost=0.1,
        )
        frame_task = FrameTask(
            "link_gripper_fingertip_right",
            position_cost=1.0,
            orientation_cost=0.1,
        )
        q = np.array(
            [
                -0.53682227,
                0.84369535,
                -1.7703322,
                0.2867902,
                0.87828404,
                0.11851416,
                0.02568168,
                0.04357896,
                0.09986985,
                -0.15279541,
                -0.39004399,
                -0.14194651,
                0.15464511,
                -0.16225863,
                0.99645724,
                0.08410094,
            ]
        )
        self.configuration.update(q)
        relative_task.set_target_from_configuration(self.configuration)
        frame_task.set_target_from_configuration(self.configuration)
        self.assertTrue(
            np.allclose(
                relative_task.compute_error(self.configuration),
                frame_task.compute_error(self.configuration),
            )
        )
        self.assertTrue(
            np.allclose(
                relative_task.compute_jacobian(self.configuration),
                frame_task.compute_jacobian(self.configuration),
            )
        )
