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

"""Lie-algebra utility functions."""

import numpy as np
import pinocchio as pin


def spatial_minus(Y: pin.SE3, X: pin.SE3) -> np.ndarray:
    r"""Compute the left minus :math:`Y \ominus_0 X`.

    The left minus operator is defined by:

    .. math::

        Y \ominus_s X = \log(Y \cdot X^{-1})

    This operator allows us to think about orientation "differences" as
    similarly as possible to position differences, but mind the frames! If we
    denote by :math:`Y = T_{0y}` and :math:`X = T_{0x}` the two transforms,
    from respectively frames :math:`y` and :math:`x` to the inertial frame
    :math:`0`, the twist resulting from a left minus is expressed in the
    inertial frame:

    .. math::

        {}_0 \xi_{0x} = Y \ominus_0 X = \log(T_{0y} T_{x0})

    A twist like :math:`{}_0 \xi_{0x}` in the inertial frame is called a
    spatial motion vector.

    Args:
        Y: Transform :math:`Y = T_{0y}` on the left-hand side of the operator.
        X: Transform :math:`X = T_{0x}` on the right-hand side of the operator.

    Returns:
        Spatial motion vector resulting from the difference :math:`\ominus_0`
        between :math:`Y` and :math:`X`.

    See also:
        The `micro Lie theory <https://arxiv.org/abs/1812.01537>`_ describes
        the difference between the left and right minus operators.
    """
    spatial_twist: np.ndarray = pin.log(Y.act(X.inverse())).vector
    return spatial_twist


def body_minus(Y: pin.SE3, X: pin.SE3) -> np.ndarray:
    r"""Compute the right minus :math:`Y \ominus X`.

    The right minus operator is defined by:

    .. math::

        Y \ominus X = \log(X^{-1} \cdot Y)

    This operator allows us to think about orientation "differences" as
    similarly as possible to position differences, but mind the frames! If we
    denote by :math:`Y = T_{0y}` and :math:`X = T_{0x}` the two transforms,
    from respectively frames :math:`y` and :math:`x` to the inertial frame
    :math:`0`, the twist resulting from a right minus is expressed in the
    local frame :math:`x`:

    .. math::

        {}_x \xi_{0x} = Y \ominus X = \log(T_{x0} T{0y}) = \log(T_{xy})

    A twist like :math:`{}_x \xi_{0x}` in the local frame :math:`x` is called a
    body motion vector.

    Args:
        Y: Transform :math:`Y = T_{0y}` on the left-hand side of the operator.
        X: Transform :math:`X = T_{0x}` on the right-hand side of the operator.

    Returns:
        Body motion vector resulting from the difference :math:`\ominus_0`
        between :math:`Y` and :math:`X`.

    See also:
        `Body motion vector
        <https://scaron.info/robotics/screw-theory.html#body-screws>`_ are also
        known as body screws in screw theory.

        The `micro Lie theory <https://arxiv.org/abs/1812.01537>`_ describes
        the difference between the left and right minus operators.
    """
    body_twist: np.ndarray = pin.log(X.actInv(Y)).vector
    return body_twist
