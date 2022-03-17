#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Stéphane Caron
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test the Configuration type.
"""

import unittest

import numpy as np

import pink

from pink.models import build_jvrc_model


class TestConfiguration(unittest.TestCase):
    def setUp(self):
        """
        Prepare test fixture.
        """
        robot = build_jvrc_model()
        self.q = robot.q0
        self.robot = robot

    def test_assume_configuration(self):
        self.robot.data.J.fill(42.0)
        configuration = pink.assume_configuration(self.robot, self.q)
        self.assertTrue(np.allclose(configuration.data.J, 42.0))

    def test_apply_configuration(self):
        self.robot.data.J.fill(42.0)
        configuration = pink.apply_configuration(self.robot, self.q)
        J_check = np.array(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
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
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
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
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
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
                    0.0,
                    0.0,
                    1.0,
                    0.0,
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
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
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
                    0.0,
                    0.0,
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
        self.assertTrue(np.allclose(configuration.data.J, J_check))

    def test_transform_found(self):
        """
        Get the pose of an existing robot body.
        """
        configuration = pink.apply_configuration(self.robot, self.q)
        transform_pelvis_to_world = configuration.get_transform_body_to_world(
            "PELVIS_S"
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
        configuration = pink.apply_configuration(self.robot, self.q)
        with self.assertRaises(KeyError):
            configuration.get_transform_body_to_world("foo")


if __name__ == "__main__":
    unittest.main()
