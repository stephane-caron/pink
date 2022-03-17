#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Stéphane Caron.
#
# This file is part of Pink.
#
# Pink is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Pink is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Pink. If not, see <http://www.gnu.org/licenses/>.

import meshcat
import numpy as np


def attach_axes(
    frame: meshcat.Visualizer,
    length: float = 0.05,
    thickness: float = 0.002,
    opacity: float = 1.0,
) -> None:
    """
    Attach a set of three basis axes to a MeshCat handle.

    Args:
        frame: MeshCat handle to attach the basis axes to.
        length: Length of axis unit vectors.
        thickness: Thickness of axis unit vectors.
        opacity: Opacity of all three unit vectors.

    Note:
        As per the de-facto standard (Blender, OpenRAVE, RViz, ...), the
        x-axis is red, the y-axis is green and the z-axis is blue.
    """
    direction_names = ["x", "y", "z"]
    colors = [0xFF0000, 0x00FF00, 0x0000FF]
    rotation_axes = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    position_cylinder_in_body = 0.5 * length * np.eye(3)
    for i in range(3):
        dir_name = direction_names[i]
        material = meshcat.geometry.MeshLambertMaterial(
            color=colors[i], opacity=opacity
        )
        transform_cylinder_to_body = meshcat.transformations.rotation_matrix(
            np.pi / 2, rotation_axes[i]
        )
        transform_cylinder_to_body[0:3, 3] = position_cylinder_in_body[i]
        cylinder = meshcat.geometry.Cylinder(length, thickness)
        frame[dir_name].set_object(cylinder, material)
        frame[dir_name].set_transform(transform_cylinder_to_body)


def set_frame(
    frame: meshcat.Visualizer,
    axis_length: float = 0.1,
    axis_thickness: float = 0.005,
    opacity: float = 1.0,
    origin_color: int = 0x000000,
    origin_radius: float = 0.01,
) -> None:
    """
    Set MeshCat handle to a frame, represented by an origin and three axes.

    Args:
        frame: MeshCat handle to attach the frame to.
        axis_length: Length of axis unit vectors, in [m].
        axis_thickness: Thickness of axis unit vectors, in [m].
        opacity: Opacity of all three unit vectors.
        origin_color: Color of the origin sphere.
        origin_radius: Radius of the frame origin sphere, in [m].

    Note:
        As per the de-facto standard (Blender, OpenRAVE, RViz, ...), the
        x-axis is red, the y-axis is green and the z-axis is blue.
    """
    material = meshcat.geometry.MeshLambertMaterial(
        color=origin_color, opacity=opacity
    )
    sphere = meshcat.geometry.Sphere(origin_radius)
    frame.set_object(sphere, material)
    attach_axes(frame, axis_length, axis_thickness, opacity)
