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
Utility classes and functions.
"""

from time import perf_counter, sleep

import numpy as np
import pinocchio as pin


def custom_configuration_vector(robot: pin.Model, **kwargs) -> np.ndarray:
    """
    Generate a configuration vector where named joints have specific values.

    Args:
        robot: Robot model.

    Returns:
        Configuration vector where named joints have the values specified in
        keyword arguments, and other joints have their neutral value.
    """
    q = pin.neutral(robot.model)
    for joint_name, joint_value in kwargs.items():
        joint_id = robot.model.getJointId(joint_name)
        joint = robot.model.joints[joint_id]
        q[joint.idx_q] = joint_value
    return q


class RateLimiter:

    """
    Regulate the frequency between calls to the same instruction in e.g. a loop
    or callback.

    This rate limiter is in essence the same as rospy.Rate_. It assumes
    Python's performance counter never jumps backward nor forward, so that it
    does not handle such cases contrary to rospy.Rate_.

    .. _rospy.Rate:
        https://github.com/ros/ros_comm/blob/noetic-devel/clients/rospy/src/rospy/timer.py

    Attributes:
        period: Desired period between two calls to :func:`sleep`, in seconds.
        slack: Duration in seconds remaining until the next tick at the
            end of the last call to :func:`sleep`.
    """

    _next_tick: float
    period: float
    slack: float

    def __init__(self, frequency: float):
        """
        Initialize rate limiter.

        Args:
            frequency: Desired frequency in hertz.
        """
        period = 1.0 / frequency
        self._next_tick = perf_counter() + period
        self.period = period

    def sleep(self):
        """
        Sleep the duration required to regulate the frequency between calls.
        """
        self.slack = self._next_tick - perf_counter()
        if self.slack > 0.0:
            sleep(self.slack)
        self._next_tick = perf_counter() + self.period
