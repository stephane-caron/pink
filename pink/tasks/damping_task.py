#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria
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

"""Damping task."""

import numpy as np

from ..configuration import Configuration
from .task import Task


class DampingTask(Task):
    r"""Minimize joint velocities.

    Damping is an analogy with forces that are proportional to velocity, and
    thus fight against motion.

    Attributes:
    """

    cost: float

    def __init__(self, cost: float) -> None:
        r"""Initialize task.

        Args:
            cost: joint angular velocity cost, in
                :math:`[\mathrm{cost}] [\mathrm{s}] / [\mathrm{rad}]`.
        """
        self.cost = cost

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        return np.zeros(configuration.model.nv)

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        return configuration.tangent.eye...
        return np.eye(configuration.model.nv)
