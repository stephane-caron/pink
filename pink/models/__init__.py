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


def build_from_urdf(urdf_path: str) -> pin.RobotWrapper:
    """
    Build a robot model from its URDF.

    Args:
        urdf_path: Path to the URDF file in a robot description folder.

    Note:
        The model to load is recognized from the name of the URDF file (e.g.
        "upkie.urdf"), or last folder in the path, for example (e.g.
        "jvrc_description").
    """
    urdf_path.rstrip("/")
    if urdf_path.endswith("jvrc1.urdf") and "jvrc_description" in urdf_path:
        return build_jvrc_from_urdf(urdf_path)
    if urdf_path.endswith("upkie.urdf") and "upkie_description" in urdf_path:
        return build_upkie_from_urdf(urdf_path)
    raise UnknownModel(urdf_path)


__all__ = [
    "build_from_urdf",
]
