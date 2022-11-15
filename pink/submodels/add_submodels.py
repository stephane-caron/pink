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
Add fields to a pinocchio.Model that are useful for IK.
"""

import pinocchio as pin

from .bounded import Bounded
from .tangent import Tangent


def add_submodels(model: pin.Model) -> None:
    """
    Extend a pinocchio.Model with configuration-independent quantities.

    Args:
        model: robot model to extend.

    Returns:
        Array of bounded joint indexes.
    """
    if not hasattr(model, "bounded"):
        model.bounded = Bounded(model)
    if not hasattr(model, "tangent"):
        model.tangent = Tangent(model)
