#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2023 Inria

"""Test fixture for other library features."""

import os
import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.iiwa14_description import PACKAGE_PATH, REPOSITORY_PATH
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink.exceptions import ConfigurationError
from pink.utils import (
    VectorSpace,
    custom_configuration_vector,
    process_collision_pairs,
)


class TestUtils(unittest.TestCase):
    """Test utility classes and functions."""

    def test_custom_configuration_vector(self):
        """Check a custom configuration vector for Upkie.

        Assumes the left and right knees have joint indices respectively 8 and
        11 in the configuration vector.
        """
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        q = custom_configuration_vector(robot, left_knee=0.2, right_knee=-0.2)
        self.assertAlmostEqual(q[8], 0.2)
        self.assertAlmostEqual(q[11], -0.2)

    def test_custom_configuration_vector_unbounded_joints(self):
        """Single number for an unbounded joint should yield an error."""
        robot = load_robot_description("gen2_description", root_joint=None)
        with self.assertRaises(ConfigurationError):
            custom_configuration_vector(robot, j2s6s200_joint_1=0.0)
        cos = np.sqrt(3) / 2.0
        sin = 0.5
        q = custom_configuration_vector(robot, j2s6s200_joint_1=[cos, sin])
        self.assertAlmostEqual(q[0], cos)
        self.assertAlmostEqual(q[1], sin)

    def test_vector_space(self):
        """Check dimensions of regular tangent space."""
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        nv = robot.model.nv
        tangent = VectorSpace(robot.model.nv)
        self.assertEqual(tangent.eye.shape, (nv, nv))
        self.assertEqual(tangent.ones.shape, (nv,))
        self.assertEqual(tangent.zeros.shape, (nv,))

    def test_process_collision_pairs(self):
        urdf_path = os.path.join(
            PACKAGE_PATH, "urdf", "iiwa14_spheres_collision.urdf"
        )
        robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path, package_dirs=[os.path.dirname(REPOSITORY_PATH)]
        )

        robot.collision_data = process_collision_pairs(
            robot.model, robot.collision_model
        )

        # The contacts are enabled
        self.assertTrue(robot.collision_data.enable_contact)
        # The amount of total collision pairs is 72
        self.assertEqual(len(robot.collision_data.distanceResults), 72)

        # If sample srdf file is taken, the number of collision pairs should
        # reduce
        robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path, package_dirs=[os.path.dirname(REPOSITORY_PATH)]
        )
        srdf_path = (
            os.path.dirname(os.path.realpath(__file__))
            + "/iiwa_exclude_pairs.srdf"
        )
        robot.collision_data = process_collision_pairs(
            robot.model, robot.collision_model, srdf_path=srdf_path
        )
        self.assertTrue(len(robot.collision_data.distanceResults) < 72)
