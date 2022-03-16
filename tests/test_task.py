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
Test base class for kinematic tasks.
"""

import unittest

from pink.models import build_jvrc_model
from pink.tasks import Task


class TestTask(unittest.TestCase):
    def setUp(self):
        """
        Prepare test fixture.
        """
        self.robot = build_jvrc_model()
        self.assertIsNotNone(self.robot)

    def test_task_not_implemented(self):
        """
        Raise an error when the robot body is not found.
        """
        task = Task()
        self.assertIsNotNone(task)
        with self.assertRaises(NotImplementedError):
            task.compute_task_dynamics(self.robot)
        with self.assertRaises(NotImplementedError):
            task.compute_qp_objective(self.robot)


if __name__ == "__main__":
    unittest.main()
