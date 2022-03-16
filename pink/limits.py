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
Joint limits implemented as inequality constraints.
"""

from typing import Tuple

import numpy as np
import pinocchio as pin


def compute_velocity_limits(
    robot: pin.RobotWrapper,
    dt: float,
    config_limit_gain: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the robot's configuration-dependent velocity limits:

    .. math::

        v_{min}(q) \\leq v \\leq v_{max}(q)

    where :math:`q \\in {\\cal C}` is the robot's configuration and :math:`v
    \\in T_q({\\cal C})` is the velocity in the tangent space at :math:`q`.
    These velocity limits combine:

    - Configuration-independent joint velocity limits
      :math:`|\\dot{q}| \\leq \\dot{q}_{lim}`.
    - Time-derivatives of the configuration limit
      :math:`q_{min} \\leq q \\leq q_{max}`.

    Args:
        robot: Robot model and configuration.
        dt: Integration timestep in [s].
        config_limit_gain: gain between 0 and 1 to steer away from
            configuration limits. It is described in "Real-time prioritized
            kinematic control under inequality constraints for redundant
            manipulators" (Kanoun, 2012). More details in [this
            writeup](https://scaron.info/teaching/inverse-kinematics.html).

    Returns:
        Pair $(v_{max}(q), v_{min}(q))$ of velocity lower and upper bounds.
    """
    v_max = +1000.0 * np.ones(robot.nv)
    v_min = -1000.0 * np.ones(robot.nv)
    return v_max, v_min
