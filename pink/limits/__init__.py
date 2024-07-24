#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

"""Limits implemented as inequality constraints in the IK problem."""

from .limit import Limit
from .configuration_limit import ConfigurationLimit
from .velocity_limit import VelocityLimit

__all__ = [
    "ConfigurationLimit",
    "Limit",
    "VelocityLimit",
]
