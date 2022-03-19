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


class MockModel:

    """
    Mock robot model, 6 joints and a floating base.
    """

    nv: int = 12


class MockConfiguration(pink.Configuration):

    """
    Mock robot configuration.
    """

    def __init__(self):
        super(MockConfiguration, self).__init__(MockModel(), None, None)
        self.transforms = {}

    def get_transform_body_to_world(self, body: str) -> pin.SE3:
        """
        Generate random poses with memoization.

        Args:
            body: Each name will return the same pose.

        Returns:
            Body pose in the world frame.
        """
        if body not in self.transforms:
            self.transforms[body] = pin.SE3.Random()
        return self.transforms[body]

    def get_body_jacobian(self, body: str) -> np.ndarray:
        return np.ones((6, 12))
