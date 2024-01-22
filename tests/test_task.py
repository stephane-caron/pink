#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Test base class for kinematic tasks."""

import unittest

from pink.tasks import Task


class TestTask(unittest.TestCase):
    """Test abstract base class for tasks."""

    def setUp(self):
        """Prepare test fixture."""
        Task.__abstractmethods__ = set()  # allow instantiation

    def test_task_repr(self):
        """String representation reports the task gain."""
        task = Task()
        self.assertTrue("gain=" in repr(task))
