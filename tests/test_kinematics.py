#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Stéphane Caron.
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
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Pink. If not, see <http://www.gnu.org/licenses/>.

"""
Test kinematics functions.
"""

import unittest
import warnings

from pink import get_transform_body_to_world

from .models import build_jvrc_model


class TestPink(unittest.TestCase):
    def setUp(self):
        """
        Prepare test fixture.
        """
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        self.robot = build_jvrc_model()

    def test_robot(self):
        self.assertIsNotNone(self.robot)

    def test_transform_not_found(self):
        with self.assertRaises(KeyError):
            get_transform_body_to_world(self.robot, "kron")


if __name__ == "__main__":
    unittest.main()
