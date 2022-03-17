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

from .configuration import Configuration
from .configuration import assume_configuration
from .configuration import apply_configuration
from .solve_ik import solve_ik

__version__ = "0.1.0"

__all__ = [
    "Configuration",
    "apply_configuration",
    "assume_configuration",
    "models",
    "solve_ik",
]
