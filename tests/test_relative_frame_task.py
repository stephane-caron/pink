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
        self.assertTrue("position_cost=" in repr(earflap_task))
        self.assertTrue("orientation_cost=" in repr(earflap_task))
        self.assertTrue("lm_damping=" in repr(earflap_task))
        self.assertTrue("gain=" in repr(earflap_task))

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
                1.0,
                0.0,
                0.99918729,
                -0.04030836,
                -0.98540495,
                0.17022655,
                -2.32040936,
                0.25954986,
                0.57941948,
                0.1000888,
                0.05202972,
                0.11589883,
                0.03683092,
                0.2766355,
                0.15045323,
                2.63148626,
                -0.51629367,
                0.53919249,
                0.9866907,
                0.16260834,
            ]
        )
        self.configuration.update(q)
        relative_task.set_target_from_configuration(self.configuration)
        frame_task.set_target_from_configuration(self.configuration)
        self.assertTrue(
            np.allclose(
                -relative_task.compute_error(self.configuration),
                frame_task.compute_error(self.configuration),
            )
        )
        self.assertTrue(
            np.allclose(
                -relative_task.compute_jacobian(self.configuration),
                frame_task.compute_jacobian(self.configuration),
            )
        )

    def test_relative_jacobian(self):
        relative_task = RelativeFrameTask(
            "link_gripper_fingertip_right",
            "universe",
            position_cost=1.0,
            orientation_cost=0.1,
        )
        q = np.array(
            [
                1.0,
                0.0,
                0.99918729,
                -0.04030836,
                -0.98540495,
                0.17022655,
                -2.32040936,
                0.25954986,
                0.57941948,
                0.1000888,
                0.05202972,
                0.11589883,
                0.03683092,
                0.2766355,
                0.15045323,
                2.63148626,
                -0.51629367,
                0.53919249,
                0.9866907,
                0.16260834,
            ]
        )
        self.configuration.update(q)
        relative_task.set_target_from_configuration(self.configuration)
        J = relative_task.compute_jacobian(self.configuration)
        J_check = np.array(
            [
                [
                    -8.90314314e-01,
                    2.62831699e-01,
                    -5.23328878e-01,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    3.63284356e-01,
                    -3.00488917e-01,
                    -3.00488917e-01,
                    -3.00488917e-01,
                    -3.00488917e-01,
                    -1.21569960e-01,
                    5.40643664e-02,
                    -2.64531107e-03,
                    -0.00000000e00,
                    -8.52051844e-02,
                    -0.00000000e00,
                ],
                [
                    -4.35448031e-01,
                    -2.52682609e-01,
                    -3.20948600e-01,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -8.66420482e-01,
                    2.59185486e-01,
                    2.59185486e-01,
                    2.59185486e-01,
                    2.59185486e-01,
                    -1.29231854e-01,
                    -1.95397417e-01,
                    -1.29331003e-01,
                    -0.00000000e00,
                    -3.40020430e-10,
                    -0.00000000e00,
                ],
                [
                    -1.33136965e-01,
                    -9.31166481e-01,
                    -2.43700772e-01,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    3.42549304e-01,
                    9.17893955e-01,
                    9.17893955e-01,
                    9.17893955e-01,
                    9.17893955e-01,
                    -1.89931943e-01,
                    1.30718019e-01,
                    -1.41489258e-03,
                    -0.00000000e00,
                    -1.71149858e-01,
                    -0.00000000e00,
                ],
                [
                    -0.00000000e00,
                    -0.00000000e00,
                    3.71833209e-01,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    3.54998241e-01,
                    7.69530324e-01,
                    -4.71641546e-01,
                    -0.00000000e00,
                    -2.81651980e-09,
                    -0.00000000e00,
                ],
                [
                    -0.00000000e00,
                    -0.00000000e00,
                    -8.64023446e-01,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -8.62832198e-01,
                    4.88273308e-01,
                    9.25233451e-10,
                    -0.00000000e00,
                    -1.00000000e00,
                    -0.00000000e00,
                ],
                [
                    -0.00000000e00,
                    -0.00000000e00,
                    3.39416483e-01,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    -0.00000000e00,
                    3.59856704e-01,
                    4.11597224e-01,
                    8.81790377e-01,
                    -0.00000000e00,
                    -2.57093808e-09,
                    -0.00000000e00,
                ],
            ]
        )
        self.assertTrue(np.allclose(J, J_check))
