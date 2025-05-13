#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Kinematic tasks."""

from .com_task import ComTask
from .damping_task import DampingTask
from .frame_task import FrameTask
from .joint_coupling_task import JointCouplingTask
from .linear_holonomic_task import LinearHolonomicTask
from .low_acceleration_task import LowAccelerationTask
from .omniwheel_task import OmniwheelTask
from .posture_task import PostureTask
from .relative_frame_task import RelativeFrameTask
from .rolling_task import RollingTask
from .task import Task

__all__ = [
    "ComTask",
    "DampingTask",
    "FrameTask",
    "JointCouplingTask",
    "LinearHolonomicTask",
    "LowAccelerationTask",
    "OmniwheelTask",
    "PostureTask",
    "RelativeFrameTask",
    "RollingTask",
    "Task",
]
