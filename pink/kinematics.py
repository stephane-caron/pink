#!/usr/bin/env python3
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
Kinematics functions.
"""

import pinocchio as pin
import pin.pinocchio_pywrap  # for pylint


def get_transform_body_to_world(
    model: pin.pinocchio_pywrap.Model,
    data: pin.pinocchio_pywrap.Data,
    body: str,
) -> pin.SE3:
    """
    Get the pose of a body frame in the current configuration.

    Args:
        model: Robot model.
        data: Data describing the current configuration of the model.
        body: Body name.

    Returns:
        Current transform from body frame to world frame.

    Raises:
        KeyError: if the body name is not found in the robot model.
    """
    body_id = model.getBodyId(body)
    try:
        return data.oMf[body_id].copy()
    except IndexError as index_error:
        raise KeyError(
            f'Body "{body}" not found in robot model'
        ) from index_error
