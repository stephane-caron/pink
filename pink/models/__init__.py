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
Collection of open source robot models.

These models are optional and not part of Pink. You can import them by
initializing and updating submodules in a local Pink repository.
"""

import pinocchio as pin

from .jvrc import build_jvrc_from_urdf
from .upkie import build_upkie_from_urdf


class UnknownModel(Exception):

    """
    Exception raised when the model to load is not recognized from its path.
    """


def build_from_urdf(path: str) -> pin.RobotWrapper:
    """
    Build a robot model from its description folder.

    Args:
        path: Path to a robot description folder.

    Note:
        The model to load is recognized from the name of the last folder in the
        path, for example "jvrc_description" or "upkie_description".
    """
    path.rstrip("/")
    if path.endswith("jvrc_description"):
        return build_jvrc_from_urdf(path)
    if path.endswith("upkie_description"):
        return build_upkie_from_urdf(path)
    raise UnknownModel(path)


__all__ = [
    "build_from_urdf",
]
