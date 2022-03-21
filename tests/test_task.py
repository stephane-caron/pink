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
Test base class for kinematic tasks.
"""

import os
import unittest

from pink.models import build_from_urdf
from pink.tasks import Task


class TestTask(unittest.TestCase):
    def setUp(self):
        """
        Prepare test fixture.
        """
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        jvrc_description = os.path.join(models_dir, "jvrc_description")
        self.jvrc_description = jvrc_description

    def test_task_not_implemented(self):
        """
        Raise an error when the robot body is not found.
        """
        robot = build_from_urdf(self.jvrc_description)
        task = Task()
        self.assertIsNotNone(task)
        with self.assertRaises(NotImplementedError):
            task.compute_task_dynamics(robot)
        with self.assertRaises(NotImplementedError):
            task.compute_qp_objective(robot)
