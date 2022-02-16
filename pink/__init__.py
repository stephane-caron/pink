#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Stéphane Caron.
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
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Pink. If not, see <http://www.gnu.org/licenses/>.

"""
Python inverse kinematics for your robot model based on Pinocchio.
"""

from .transforms import get_transform_body_to_world

__version__ = "0.0.2rc1"

__all__ = [
    "get_transform_body_to_world",
]
