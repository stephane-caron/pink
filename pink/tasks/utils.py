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


def box_minus(plus_transform: pin.SE3, minus_transform: pin.SE3) -> np.ndarray:
    r"""Compute the box minus between two transforms.

    The box minus operator is defined by:

    .. math::

        T_1 \boxminus T_2 = \log(T_1 \cdot T_2^{-1})

    This operator allows us to think about orientation "differences" as
    similarly as possible to position differences, but mind the frames! Its
    formula has two use cases, depending on whether the common frame :math:`C`
    between the two transforms is their source or their target.

    When the common frame is the target, denoting by :math:`T_{CP}` the
    transform from frame :math:`P` (source) to frame :math:`C` (target), the
    resulting twist is expressed in the target frame:

    .. math::

        {}_C \xi_{CM} = T_{CP} \boxminus T_{CM}

    When the common frame is the source frame, denoting by :math:`T_{MC}` the
    transform from frame :math:`C` (source) to frame :math:`M` (target), the
    resulting twist is expressed in the target frame of the transform on the
    right-hand side of the operator:

    .. math::

        -{}_M \xi_{M} = T_{PC} \boxminus T_{MC}

    Args:
        plus_transform: Transform :math:`T_1` on the left-hand side of the box
            minus operator.
        minus_transform: Transform :math:`T_2` on the right-hand side of the
            box minus operator.

    Returns:
        In the first case :math:`T_{CP} \boxminus T_{CM}`, the outcome is a
        spatial twist :math:`{}_C \xi_{CM}` expressed in the common frame
        :math:`C`.

        In the second case :math:`T_{PC} \boxminus T_{MC}`, the outcome is a
        body twist :math:`-{}_M \xi_{CM}` (mind the unitary minus).

    Note:
        Prefer using :func:`pink.tasks.utils.body_box_minus` to calling this
        function in the second use case :math:`T_{PC} \boxminus T_{MC}`.
    """
    diff_array = plus_transform.act(minus_transform.inverse())
    twist: np.ndarray = pin.log(diff_array).vector
    return twist


def body_box_minus(
    transform_plus_to_common: pin.SE3,
    transform_minus_to_common: pin.SE3,
) -> np.ndarray:
    r"""Compute the body variant of the box minus.

    The body error between two poses :math:`P` (plus) and :math:`M` (minus),
    represented by their transforms :math:`T_{CP}` and :math:`T_{CM}` to a
    common frame :math:`C`, is defined in Pink by:

    .. math::

        {}_M \xi_{CM} = -(T_{PC} \boxminus T_{MC})
                       = -(T_{CP}^{-1} \boxminus T_{CM}^{-1})

    Args:
        transform_plus_to_common: Transform :math:`T_{CP}` from the "plus"
            frame to the common frame.
        transform_minus_to_common: Transform :math:`T_{CM}` from the "minus"
            frame to the common frame.

    Returns:
        Body twist :math:`{}_M \xi_{CM}`. If :math:`C` is inertial (and has
        zero velocity) we can simply write it :math:`{}_M \xi_M` like other
        `body screws
        <https://scaron.info/robotics/screw-theory.html#body-screws>`_.
    """
    return box_minus(
        transform_plus_to_common.inverse(),
        transform_minus_to_common.inverse(),
    )
