#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

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
        print("If you `pip install yourdfpy`, this example will display it.")
    else:  # yourdfpy is not None
        viz = yourdfpy.URDF.load(urdf_path)
        viz.show()
