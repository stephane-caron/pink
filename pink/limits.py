#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Stéphane Caron
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

"""
Joint limits implemented as inequality constraints.
"""

from typing import Tuple

import numpy as np

from .configuration import Configuration
from .utils import get_root_joint_dim


def compute_velocity_limits(
    configuration: Configuration,
    dt: float,
    config_limit_gain: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the configuration-dependent velocity limits:

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
        configuration: Robot configuration to read kinematics from.
        dt: Integration timestep in [s].
        config_limit_gain: gain between 0 and 1 to steer away from
            configuration limits. It is described in "Real-time prioritized
            kinematic control under inequality constraints for redundant
            manipulators" (Kanoun, 2012). More details in [this
            writeup](https://scaron.info/teaching/inverse-kinematics.html).

    Returns:
        Pair $(v_{max}(q), v_{min}(q))$ of velocity lower and upper bounds.
    """
    assert 0.0 < config_limit_gain <= 1.0

    # Velocity limits from URDF
    v_max = configuration.model.velocityLimit.copy()
    if v_max.dot(v_max) < 1e-10:
        # Zero means no limit, see https://wiki.ros.org/urdf/XML/link
        v_max = np.full(v_max.shape, +np.infty)
    v_min = -v_max

    # Configuration limits, only defined for actuated joints
    root_nq, root_nv = get_root_joint_dim(configuration.model)
    q_act = configuration.q[root_nq:]
    q_max = configuration.model.upperPositionLimit[root_nq:]
    q_min = configuration.model.lowerPositionLimit[root_nq:]
    no_limit = q_max <= q_min + 1e-10
    q_max[no_limit] = np.inf
    q_min[no_limit] = -np.inf

    # Apply both limits
    v_max[root_nv:] = np.minimum(
        v_max[root_nv:], config_limit_gain * (q_max - q_act) / dt
    )
    v_min[root_nv:] = np.maximum(
        v_min[root_nv:], config_limit_gain * (q_min - q_act) / dt
    )

    return v_max, v_min
