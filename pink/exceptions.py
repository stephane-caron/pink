#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2024 Inria

"""Exceptions specific to Pink."""


class PinkError(Exception):
    """Base class for Pink exceptions."""


class BodyNotFound(PinkError):
    """Exception raised when a body is not found in the robot model."""


class ConfigurationError(PinkError):
    """Exception raised when encountering an invalid configuration vector."""


class NotWithinConfigurationLimits(PinkError):
    """Exception thrown when a robot configuration violates its limits.

    Attributes:
        joint: Index of the joint in the configuration vector.
        value: Invalid value of the joint.
        lower: Minimum allowed value for this joint.
        upper: Maximum allowed value for this joint.
    """

    joint: int
    value: float
    lower: float
    upper: float

    def __init__(
        self,
        joint: int,
        value: float,
        lower: float,
        upper: float,
    ) -> None:
        """Create exception.

        Args:
            joint: Index of the joint in the configuration vector.
            value: Invalid value of the joint.
            lower: Minimum allowed value for this joint.
            upper: Maximum allowed value for this joint.
        """
        self.joint = joint
        self.value = value
        self.lower = lower
        self.upper = upper
        self.message = (
            f"Joint {joint} violates configuration limits "
            f"{lower} <= {value} <= {upper}"
        )
        super().__init__(self.message)
