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
