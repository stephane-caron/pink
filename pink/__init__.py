#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2024 Inria

"""Inverse kinematics for articulated robot models, based on Pinocchio."""

from .configuration import Configuration
from .exceptions import PinkError
from .solve_ik import build_ik, solve_ik
from .tasks import (
    FrameTask,
    JointCouplingTask,
    JointVelocityTask,
    LinearHolonomicTask,
    PostureTask,
    Task,
)
from .utils import custom_configuration_vector

__version__ = "3.4.0"

__all__ = [
    "Configuration",
    "FrameTask",
    "JointCouplingTask",
    "JointVelocityTask",
    "LinearHolonomicTask",
    "PinkError",
    "PostureTask",
    "Task",
    "build_ik",
    "custom_configuration_vector",
    "solve_ik",
]
