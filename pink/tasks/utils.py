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

"""
Lie algebra utility functions.
"""

import numpy as np
import pinocchio as pin


def box_minus(plus_transform: pin.SE3, minus_transform: pin.SE3) -> np.ndarray:
    """
    Compute the box minus between two transforms:

    .. math::

        T_1 \\boxminus T_2 = \\log(T_1 \\cdot T_2^{-1})

    This operator allows us to think about orientation "differences" as
    similarly as possible to position differences, but mind the frames! Its
    formula has two use cases, depending on whether the common frame :math:`C`
    between the two transforms is their source or their target.

    When the common frame is the target, denoting by :math:`T_{CP}` the
    transform from frame :math:`P` (source) to frame :math:`C` (target), the
    resulting twist is expressed in the target frame:

    .. math::

        {}_C \\xi_{CM} = T_{CP} \\boxminus T_{CM}

    When the common frame is the source frame, denoting by :math:`T_{MC}` the
    transform from frame :math:`C` (source) to frame :math:`M` (target), the
    resulting twist is expressed in the target frame of the transform on the
    right-hand side of the operator:

    .. math::

        -{}_M \\xi_{M} = T_{PC} \\boxminus T_{MC}

    Args:
        plus_transform: Transform :math:`T_1` on the left-hand side of the box
            minus operator.
        minus_transform: Transform :math:`T_2` on the right-hand side of the
            box minus operator.

    Returns:
        In the first case :math:`T_{CP} \\boxminus T_{CM}`, the outcome is a
        spatial twist :math:`{}_C \\xi_{CM}` expressed in the common frame
        :math:`C`.

        In the second case :math:`T_{PC} \\boxminus T_{MC}`, the outcome is a
        body twist :math:`-{}_M \\xi_{CM}` (mind the unitary minus).

    Note:
        Prefer using :func:`pink.tasks.utils.body_box_minus` to calling this
        function in the second use case :math:`T_{PC} \\boxminus T_{MC}`.
    """
    diff_array = plus_transform.act(minus_transform.inverse())
    twist: np.ndarray = pin.log(diff_array).vector
    return twist


def body_box_minus(
    transform_plus_to_common: pin.SE3,
    transform_minus_to_common: pin.SE3,
) -> np.ndarray:
    """
    Compute the box minus between two poses :math:`P` (plus) and :math:`M`
    (minus) represented by their transforms :math:`T_{CP}` and :math:`T_{CM}`
    to a common frame :math:`C`.

    .. math::

        {}_M \\xi_{CM} = -(T_{PC} \\boxminus T_{MC})
                       = -(T_{CP}^{-1} \\boxminus T_{CM}^{-1})

    Args:
        transform_plus_to_common: Transform :math:`T_{CP}` from the "plus"
            frame to the common frame.
        transform_minus_to_common: Transform :math:`T_{CM}` from the "minus"
            frame to the common frame.

    Returns:
        Body twist :math:`{}_M \\xi_{CM}`. If :math:`C` is inertial (and has
        zero velocity) we can simply write it :math:`{}_M \\xi_M` like other
        `body screws
        <https://scaron.info/robotics/screw-theory.html#body-screws>`_.
    """
    return box_minus(
        transform_plus_to_common.inverse(),
        transform_minus_to_common.inverse(),
    )


def get_transform_body_to_world(
    robot: pin.RobotWrapper,
    body: str,
) -> pin.SE3:
    """
    Get the pose of a body frame of the robot in its current configuration.

    Args:
        robot: Robot model and data describing its current configuration.
        body: Body name.

    Returns:
        Current transform from body frame to world frame.

    Raises:
        KeyError: if the body name is not found in the robot model.
    """
    body_id = robot.model.getBodyId(body)
    try:
        return robot.data.oMf[body_id].copy()
    except IndexError as index_error:
        raise KeyError(
            f'Body "{body}" not found in robot model'
        ) from index_error


def compute_body_jacobian(
    robot: pin.RobotWrapper, configuration: np.ndarray, body_name: str
) -> np.ndarray:
    """
    Compute the Jacobian matrix :math:`{}_B J_{WB}` of the body velocity
    :math:`{}_B v_{WB}`:

    .. math::

        {}_B v_{WB} = {}_B J_{WB} \\dot{q}

    Args:
        robot: Robot model and data describing its current configuration.
        configuration: Vector of joint coordinates.
        body_name: Name of robot body.

    Returns:
        Jacobian :math:`{}_B J_{WB}` of the body twist.

    When the robot model includes a floating base (pin.JointModelFreeFlyer),
    the configuration vector :math:`q` consists of:

    - ``q[0:3]``: position in [m] of the floating base in the inertial frame,
      formatted as :math:`[p_x, p_y, p_z]`.
    - ``q[3:7]``: unit quaternion for the orientation of the floating base in
      the inertial frame, formatted as :math:`[q_x, q_y, q_z, q_w]`.
    - ``q[7:]``: joint angles in [rad].
    """
    body_id = robot.model.getBodyId(body_name)
    J: np.ndarray = robot.computeFrameJacobian(configuration, body_id)
    return J
