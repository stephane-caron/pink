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
Test fixture for other library features.
"""

import unittest

from pink.models import UnknownModel
from pink.models import build_from_urdf


class TestConfiguration(unittest.TestCase):
    def test_build_from_unknown_model(self):
        """
        Raise exception when model is not found.
        """
        with self.assertRaises(UnknownModel):
            build_from_urdf("foo_smash_super_bar")
