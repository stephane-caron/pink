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
Subspace of the configuration or tangent space.
"""

import numpy as np


class Subspace:

    """
    Subspace of an input space (configuration or tangent) restricted to a given
    set of joints.

    Attributes:
        dim: Dimension of output space.
        indices: Joint coordinates in input space.
        input_dim: Dimension of input space.
        projection_matrix: Projection matrix from input to output space.
    """

    dim: int
    indices: np.ndarray
    input_dim: int
    projection_matrix: np.ndarray

    def __init__(self, input_dim: int, indices: np.ndarray):
        if len(indices) < 1:
            raise ValueError("cannot project on empty space")
        projection_matrix = np.eye(input_dim)[indices]
        self.dim = len(indices)
        self.indices = indices
        self.input_dim = input_dim
        self.projection_matrix = projection_matrix

    def project(self, v: np.ndarray) -> np.ndarray:
        """
        Project a vector from the original space to the subspace.

        Args:
            v: Vector from the original space.
        """
        assert v.shape == (self.input_dim,), "Dimension mismatch"
        return v[self.indices]
