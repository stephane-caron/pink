#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Visualization helpers."""

import pinocchio as pin


def start_meshcat_visualizer(
    robot: pin.RobotWrapper,
) -> pin.visualize.MeshcatVisualizer:
    """Open a MeshCat visualizer in a web browser.

    Args:
        robot: Robot wrapper with its initial data.

    Returns:
        MeshCat visualizer.
    """
    viz = pin.visualize.MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model
    )
    robot.setVisualizer(viz, init=False)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    return viz
