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

"""All kinematic limits derive from the :class:`Limit` base class."""

import abc
from typing import Optional, Tuple

import numpy as np


class Limit(abc.ABC):
    r"""Abstract base class for kinematic limits."""

    @abc.abstractmethod
    def compute_qp_inequalities(
        self,
        q: np.ndarray,
        dt: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        r"""Compute limit as linearized QP inequalities.

        Those limits are returned as:

        .. math::

            G(q) \Delta q \leq h(q)

        where :math:`q \in {\cal C}` is the robot's configuration and
        :math:`\Delta q \in T_q({\cal C})` is the displacement in the tangent
        space at :math:`q`.

        Args:
            q: Robot configuration.
            dt: Integration timestep in [s].

        Returns:
            Pair :math:`(G, h)` representing the inequality constraint as
            :math:`G \Delta q \leq h`, or ``None`` if there is no limit.
        """
