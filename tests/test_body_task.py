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
Test fixture for the body task.
"""

import os
import unittest

import numpy as np

from pink.tasks import BodyTask
from pink.tasks import TargetNotSet

from .mock_configuration import MockConfiguration


class TestBodyTask(unittest.TestCase):

    """
    Test consistency of the body task.

    Note:
        This fixture only tests the task itself. Integration tests with the IK
        are carried out in :class:`TestSolveIK`.
    """

    def setUp(self):
        """
        Prepare test fixture.
        """
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        jvrc_description = os.path.join(models_dir, "jvrc_description")
        self.jvrc_description = jvrc_description
        self.mock_configuration = MockConfiguration()

    def test_target_not_set(self):
        jetpack_task = BodyTask(
            "jetpack", position_cost=1.0, orientation_cost=0.1
        )
        with self.assertRaises(TargetNotSet):
            jetpack_task.compute_task_dynamics(self.mock_configuration)

    def test_target_set_properly(self):
        jetpack_task = BodyTask(
            "jetpack", position_cost=1.0, orientation_cost=0.1
        )
        T = self.mock_configuration.get_transform_body_to_world("jetpack")
        jetpack_task.set_target(T)
        self.assertIsNotNone(jetpack_task.transform_target_to_world)
        if jetpack_task.transform_target_to_world is not None:  # help mypy
            self.assertTrue(
                np.allclose(
                    T.homogeneous,
                    jetpack_task.transform_target_to_world.homogeneous,
                )
            )


if __name__ == "__main__":
    unittest.main()
