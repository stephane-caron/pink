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

"""Exceptions specific to Pink."""

class PinkError(Exception):
    pass


class BodyNotFound(PinkError):
    pass


class NotWithinConfigurationLimits(PinkError):

    """
    Exception thrown when a robot configuration violates its limits.

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
        """
        Create exception.

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
