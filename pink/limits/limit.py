#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

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
