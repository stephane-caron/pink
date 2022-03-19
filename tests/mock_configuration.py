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
Mock configuration used by test fixtures.
"""

import numpy as np
import pinocchio as pin

import pink


class MockConfiguration(pink.Configuration):

    """
    Mock robot configuration.
    """

    def __init__(self):
        super(MockConfiguration, self).__init__(None, None, None)

    def get_transform_body_to_world(self, body: str) -> pin.SE3:
        return pin.SE3.Random()

    def get_body_jacobian(self, body: str) -> np.ndarray:
        return np.ones((6, 12))
