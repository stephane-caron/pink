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
Tangent space associated with a robot model.
"""

import numpy as np
import pinocchio as pin


class Tangent:

    """
    Represent the tangent space of a pinocchio.Model.

    Attributes:
        bounded_proj: Projection matrix on bounded joints.
        eye: Identity matrix.
        ones: Tangent space vector with all coordinates at one.
        zeros: Zero tangent space vector.
    """

    eye: np.ndarray
    ones: np.ndarray
    zeros: np.ndarray

    def __init__(self, model: pin.Model):
        eye = np.eye(model.nv)
        ones = np.ones(model.nv)
        zeros = np.zeros(model.nv)
        self.eye = eye
        self.ones = ones
        self.zeros = zeros
