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

import unittest

import numpy as np

from pink import apply_configuration, solve_ik
from pink.models import build_jvrc_model
from pink.tasks import BodyTask


class TestBodyTask(unittest.TestCase):
    def setUp(self):
        """
        Prepare test fixture.
        """
        self.dt = 5e-3  # [s]
        self.robot = build_jvrc_model()
        self.assertIsNotNone(self.robot)

    def test_task_fulfilled(self):
        """
        No motion when all targets are reached.
        """
        configuration = apply_configuration(self.robot, self.robot.q0)
        left_ankle_task = BodyTask(
            "l_ankle", position_cost=1.0, orientation_cost=3.0
        )
        right_ankle_task = BodyTask(
            "r_ankle", position_cost=1.0, orientation_cost=3.0
        )
        pelvis_task = BodyTask(
            "PELVIS_S", position_cost=1.0, orientation_cost=3.0
        )

        left_ankle_task.set_target(
            configuration.get_transform_body_to_world("l_ankle")
        )
        right_ankle_task.set_target(
            configuration.get_transform_body_to_world("r_ankle")
        )
        pelvis_task.set_target(
            configuration.get_transform_body_to_world("PELVIS_S")
        )

        tasks = [pelvis_task, left_ankle_task, right_ankle_task]
        velocity = solve_ik(configuration, tasks, self.dt)
        self.assertTrue(np.allclose(velocity, 0.0))


if __name__ == "__main__":
    unittest.main()
