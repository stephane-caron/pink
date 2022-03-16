#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Stéphane Caron.
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
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Pink. If not, see <http://www.gnu.org/licenses/>.

"""
Python inverse kinematics for your robot model based on Pinocchio.
"""

from .configured_robot import ConfiguredRobot
from .configured_robot import assume_robot_is_configured
from .configured_robot import configure_robot
from .solve_ik import solve_ik

__version__ = "0.0.2"

__all__ = [
    "ConfiguredRobot",
    "assume_robot_is_configured",
    "configure_robot",
    "models",
    "solve_ik",
]
