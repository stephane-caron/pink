#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Visualization helpers."""

import pinocchio as pin
from pinocchio import visualize

try:
    import hppfcl

    MESH_TYPES = (
        hppfcl.BVHModelBase,
        hppfcl.HeightFieldOBBRSS,
        hppfcl.HeightFieldAABB,
    )
    HAS_HPPFCL = True
except ImportError:
    HAS_HPPFCL = False
    MESH_TYPES = ()


def _patched_load_viewer_geometry_object(
    self, geometry_object, prefix="", color=None
):
    """Patched version that preserves mesh colors when color is None.

    The original implementation always uses geometry_object.meshColor as
    fallback, which overrides the actual mesh file colors with a default
    white color. This patched version passes color=None to _add_mesh_from_path
    for mesh types when no explicit color override is specified, allowing
    trimesh to load the original mesh colors.

    Note: This patch will not be required once pinocchio 3.9.0 is released
        to PyPi, which includes a fix forthis issue:
        https://github.com/stack-of-tasks/pinocchio/releases/tag/v3.9.0
    """
    import trimesh

    name = geometry_object.name
    if prefix:
        name = prefix + "/" + name
    geom = geometry_object.geometry
    # Only use meshColor fallback for primitive shapes, not for mesh files
    color_override = color if color is not None else geometry_object.meshColor

    if isinstance(geom, hppfcl.Box):
        frame = self.viewer.scene.add_box(
            name,
            dimensions=geom.halfSide * 2.0,
            color=color_override[:3],
            opacity=color_override[3],
        )
    elif isinstance(geom, hppfcl.Sphere):
        frame = self.viewer.scene.add_icosphere(
            name,
            radius=geom.radius,
            color=color_override[:3],
            opacity=color_override[3],
        )
    elif isinstance(geom, hppfcl.Cylinder):
        mesh = trimesh.creation.cylinder(
            radius=geom.radius,
            height=geom.halfLength * 2.0,
        )
        frame = self.viewer.scene.add_mesh_simple(
            name,
            mesh.vertices,
            mesh.faces,
            color=color_override[:3],
            opacity=color_override[3],
        )
    elif isinstance(geom, MESH_TYPES):
        # Pass original color (may be None) to preserve mesh file colors
        frame = self._add_mesh_from_path(name, geometry_object.meshPath, color)
    elif isinstance(geom, hppfcl.Convex):
        if len(geometry_object.meshPath) > 0:
            # Pass original color (may be None) to preserve mesh file colors
            frame = self._add_mesh_from_path(
                name, geometry_object.meshPath, color
            )
        else:
            frame = self._add_mesh_from_convex(name, geom, color_override)
    else:
        raise RuntimeError(
            f"Unsupported geometry type for {name}: {type(geom)}"
        )

    self.frames[name] = frame


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
    if pin.__version__ < "3.8.0":
        raise RuntimeError(
            f"Viser visualizer requires pinocchio 3.8.0 or later,"
            f"but found {pin.__version__}"
        )

    visualizer = visualize.ViserVisualizer(
        robot.model, robot.collision_model, robot.visual_model
    )

    # HACK: Patch the loadViewerGeometryObject method to preserve mesh colors
    # The original pinocchio implementation always falls back to meshColor
    # (typically white) even for mesh files that have their own colors/textures
    # This patch will not be required once pinocchio 3.9.0 is released to PyPi,
    # which includes a fix for this issue:
    # https://github.com/stack-of-tasks/pinocchio/releases/tag/v3.9.0
    if HAS_HPPFCL:
        import types

        visualizer.loadViewerGeometryObject = types.MethodType(
            _patched_load_viewer_geometry_object, visualizer
        )

    robot.setVisualizer(visualizer, init=False)
    visualizer.initViewer(open=open)
    visualizer.loadViewerModel()
    visualizer.displayCollisions(False)
    visualizer.displayVisuals(True)
    visualizer.displayFrames(False)
    return visualizer
