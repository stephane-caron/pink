#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Exceptions raised by tasks."""

from ..exceptions import PinkError


class TargetNotSet(PinkError):
    """Exception raised when attempting to compute with an unset target."""


class TaskDefinitionError(PinkError):
    """Exception raised when a task definition is ill-formed."""


class TaskJacobianNotSet(PinkError):
    """Exception raised when attempting to compute without a task Jacobian."""
