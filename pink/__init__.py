#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Python inverse kinematics for your robot model based on Pinocchio."""

from .configuration import Configuration
from .solve_ik import build_ik, solve_ik
from .tasks import Task
from .utils import custom_configuration_vector

__version__ = "1.0.0"

__all__ = [
    "Configuration",
    "build_ik",
    "custom_configuration_vector",
    "solve_ik",
    "Task",
]
