#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Lie-algebra utility functions."""

import numpy as np
import pinocchio as pin


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

    Notes:
        - `Body motion vector
          <https://scaron.info/robotics/screw-theory.html#body-screws>`_ are
          also known as body screws in screw theory.
        - The `micro Lie theory <https://arxiv.org/abs/1812.01537>`_ describes
          the difference between the left and right minus operators.
    """
    body_twist: np.ndarray = pin.log(X.actInv(Y)).vector
    return body_twist
