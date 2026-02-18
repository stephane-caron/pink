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
    open: bool = True,  # pylint: disable=redefined-builtin
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


def start_viser_visualizer(
    robot: pin.RobotWrapper,
    open: bool = True,
) -> visualize.ViserVisualizer:
    """Open a Viser visualizer in a Web browser.

    Args:
        robot: Pinocchio robot wrapper with its model and data.
        open: If set (default), open Viser in a new Web browser tab.

    Returns:
        Viser visualizer.
    """
    if pin.__version__ < "3.9.0":
        raise RuntimeError(
            f"Viser visualizer requires pinocchio 3.9.0 or later,"
            f"but found {pin.__version__}"
        )

    visualizer = visualize.ViserVisualizer(
        robot.model, robot.collision_model, robot.visual_model
    )

    robot.setVisualizer(visualizer, init=False)
    visualizer.initViewer(open=open)
    visualizer.loadViewerModel()
    visualizer.displayCollisions(False)
    visualizer.displayVisuals(True)
    visualizer.displayFrames(False)
    return visualizer
