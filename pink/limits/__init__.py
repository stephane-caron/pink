#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

r"""Limits implemented as inequality constraints in the IK problem.

Kinematic limits derive from the :class:`Limit` base class. They compute QP
inequality constraints of the form:

.. math::

    G(q) \Delta q \leq h(q)

where :math:`q \in {\cal C}` is the robot's configuration and
:math:`\Delta q \in T_q({\cal C})` is the displacement in the tangent
space at :math:`q`.
"""

from .acceleration_limit import AccelerationLimit
from .configuration_limit import ConfigurationLimit
from .limit import Limit
from .velocity_limit import VelocityLimit

__all__ = [
    "AccelerationLimit",
    "ConfigurationLimit",
    "Limit",
    "VelocityLimit",
]
