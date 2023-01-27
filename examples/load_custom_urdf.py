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

"""Load a custom URDF."""

import os

import pinocchio as pin

try:
    import yourdfpy
except ImportError:
    yourdfpy = None

if __name__ == "__main__":
    urdf_path = os.path.join(
        os.path.dirname(__file__),
        "robots",
        "double_pendulum.urdf",
    )

    robot = pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=["."],
        root_joint=None,
    )
    print(f"URDF description successfully loaded in {robot}")

    if yourdfpy is None:
        print("If you ``pip install yourdfpy``, this example will display it.")
    else:  # yourdfpy is not None
        viz = yourdfpy.URDF.load(urdf_path)
        viz.show()
