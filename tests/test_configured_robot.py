#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Stéphane Caron.
#
# This file is part of Pink.
#
# Pink is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Pink is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Pink. If not, see <http://www.gnu.org/licenses/>.

"""
Test the ConfiguredRobot type.
"""

import unittest

import numpy as np

import pink

from pink.models import build_jvrc_model


class TestSolveIK(unittest.TestCase):
    def setUp(self):
        """
        Prepare test fixture.
        """
        robot = build_jvrc_model()
        self.q = robot.q0
        self.robot = robot

    def test_configure_robot(self):
        self.robot.data.J.fill(42.0)
        assumed_robot = pink.assume_robot_is_configured(self.robot)
        self.assertTrue(np.allclose(assumed_robot.data.J, 42.0))
        configured_robot = pink.configure_robot(self.robot, self.q)
        J_check = np.array(
            [
                [
                    0.0,
                    0.0,
                    0.096,
                    0.389,
                    0.0,
                    0.746,
                    -0.0,
                    -0.0,
                    -0.096,
                    0.389,
                    0.0,
                    0.746,
                    0.0,
                    -0.192,
                    0.0,
                    -0.522,
                    0.0,
                    0.24,
                    -0.217,
                    0.24,
                    0.0,
                    0.24,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.645,
                    -0.522,
                    -0.0,
                    -0.24,
                    -0.217,
                    -0.24,
                    0.0,
                    -0.24,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.746,
                    -0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.746,
                    -0.0,
                    0.0,
                    0.0,
                    0.192,
                    0.0,
                    0.522,
                    0.0,
                    0.0,
                    -0.0,
                    -0.022,
                    -0.0,
                    -0.124,
                    -0.169,
                    -0.124,
                    -0.169,
                    -0.106,
                    -0.151,
                    0.003,
                    0.645,
                    0.0,
                    0.0,
                    0.522,
                    0.0,
                    0.0,
                    -0.0,
                    -0.022,
                    -0.0,
                    -0.124,
                    -0.169,
                    -0.124,
                    -0.169,
                    -0.106,
                    -0.151,
                ],
                [
                    0.0,
                    -0.096,
                    0.0,
                    -0.02,
                    -0.096,
                    0.02,
                    0.0,
                    0.096,
                    0.0,
                    -0.02,
                    0.096,
                    0.02,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.24,
                    0.0,
                    0.004,
                    0.0,
                    -0.24,
                    0.0,
                    -0.246,
                    -0.246,
                    -0.246,
                    -0.246,
                    -0.217,
                    -0.217,
                    -0.0,
                    -0.0,
                    -0.003,
                    0.0,
                    0.24,
                    0.0,
                    0.004,
                    0.0,
                    0.24,
                    0.0,
                    0.246,
                    0.246,
                    0.246,
                    0.246,
                    0.217,
                    0.217,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                [
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        )
        self.assertTrue(np.allclose(configured_robot.data.J, J_check))

    def test_transform_found(self):
        """
        Get the pose of an existing robot body.
        """
        configured_robot = pink.configure_robot(self.robot, self.q)
        transform_pelvis_to_world = (
            configured_robot.get_transform_body_to_world("PELVIS_S")
        )
        self.assertTrue(
            np.allclose(
                transform_pelvis_to_world.np[3, :],
                np.array([0.0, 0.0, 0.0, 1.0]),
            )
        )

    def test_transform_not_found(self):
        """
        Raise an error when the robot body is not found.
        """
        configured_robot = pink.configure_robot(self.robot, self.q)
        with self.assertRaises(KeyError):
            configured_robot.get_transform_body_to_world("foo")


if __name__ == "__main__":
    unittest.main()
