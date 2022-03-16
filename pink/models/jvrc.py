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
JVRC-1 humanoid robot model.
"""

import os

import pinocchio as pin


def build_jvrc_model():
    models_dir = os.path.join(os.path.dirname(__file__))
    urdf_path = os.path.join(
        models_dir, "jvrc_description", "urdf", "jvrc1.urdf"
    )
    return pin.RobotWrapper.BuildFromURDF(urdf_path, models_dir)
