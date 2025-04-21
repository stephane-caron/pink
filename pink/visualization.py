#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Visualization helpers."""

import pinocchio as pin
from pinocchio import visualize


def start_meshcat_visualizer(
    robot: pin.RobotWrapper,
    open: bool = True,
) -> visualize.MeshcatVisualizer:
    """Open a MeshCat visualizer in a Web browser.

    Args:
        robot: Pinocchio robot wrapper with its model and data.
        open: If set (default), open MeshCat in a new Web browser tab.

    Returns:
        MeshCat visualizer.
    """
    visualizer = visualize.MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model
    )
    robot.setVisualizer(visualizer, init=False)
    visualizer.initViewer(open=open)
    visualizer.loadViewerModel()
    return visualizer
