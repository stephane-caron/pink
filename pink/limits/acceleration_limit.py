#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Inria

"""Subset of acceleration-limited joints in a robot model."""

from typing import List, Optional, Tuple

import numpy as np
import pinocchio as pin

from ..exceptions import PinkError
from .limit import Limit


class AccelerationLimit(Limit):
    r"""Subset of acceleration-limited joints in a robot model.

    Acceleration limits consists of two parts: the expected :math:`|a| \leq
    a_{\mathrm{max}}`, but also the following term accounting for the "breaking
    distance" to configuration limits:

    .. math::

        -\sqrt{2 a_max (q \ominus q_{\min})} \leq a \leq \sqrt{2 a_max
        (q_{\max} \ominus q)}

    This additional inequality is detailed in [Flacco2015]_ as well as in
    [DelPrete2018]_.

    Attributes:
        Delta_q_prev: Latest displacement of the robot.
        a_max: Maximum acceleration vector for acceleration-limited joints.
        indices: Tangent indices corresponding to acceleration-limited joints.
        model: Robot model.
        projection_matrix: Projection from tangent space to subspace with
            acceleration-limited joints.
    """

    Delta_q_prev: np.ndarray
    a_max: np.ndarray
    indices: np.ndarray
    model: pin.Model
    projection_matrix: Optional[np.ndarray]

    def __init__(self, model: pin.Model, acceleration_limit: np.ndarray):
        """Initialize bounded tangent of a model.

        Args:
            model: robot model.
            acceleration_limit: Vector of acceleration limits, of dimension
                ``model.nv``.
        """
        acceleration_limit = acceleration_limit.flatten()
        if model.nv > 0 and acceleration_limit.shape[0] != model.nv:
            raise PinkError(f"{acceleration_limit.shape=} but {model.nv=}")

        has_acceleration_limit = np.logical_and(
            acceleration_limit < 1e20,
            acceleration_limit > 1e-10,
        )

        joints = [
            joint
            for joint in model.joints
            if joint.idx_v >= 0
            and has_acceleration_limit[
                slice(joint.idx_v, joint.idx_v + joint.nv)
            ].all()
        ]

        index_list: List[int] = []
        for joint in joints:
            index_list.extend(range(joint.idx_v, joint.idx_v + joint.nv))
        indices = np.array(index_list)
        indices.setflags(write=False)

        dim = len(indices)
        projection_matrix = np.eye(model.nv)[indices] if dim > 0 else None
        a_max = acceleration_limit[indices] if dim > 0 else np.empty(0)

        self.Delta_q_prev = np.zeros(dim)
        self.a_max = a_max
        self.indices = indices
        self.model = model
        self.projection_matrix = projection_matrix

    def set_last_integration(self, v_prev: np.ndarray, dt) -> None:
        """Set the latest velocity and the duration it was applied for.

        The goal of the low-acceleration task is to minimize the difference
        between the new velocity and the previous one.

        Args:
            v_prev: Latest integrated velocity.
            dt: Integration timestep in [s].
        """
        Delta_q_prev_full = v_prev * dt
        self.Delta_q_prev = Delta_q_prev_full[self.indices]

    def compute_qp_inequalities(
        self,
        q: np.ndarray,
        dt: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        r"""Compute inequalities for acceleration limits.

        Those limits are defined by:

        .. math::

            \Delta q_{\mathrm{prev}} - a_max \mathrm{d} t^2
            \leq \Delta q
            \leq \Delta q_{\mathrm{prev}} + a_max \mathrm{d} t^2

        where :math:`a_{max} \in {\cal T}` is the robot's acceleration limit
        vector (in the tangent space) and :math:`\Delta q \in T_q({\cal C})` is
        the displacement computed by the inverse kinematics, with :math:`\Delta
        q_{\mathrm{prev}}` the displacement from the previous iteration.

        Args:
            q: Robot configuration.
            dt: Integration timestep in [s].

        Returns:
            Pair :math:`(G, h)` representing the inequality constraint as
            :math:`G \Delta q \leq h`, or ``None`` if there is no limit.
        """
        if self.projection_matrix is None:  # no joint (thus checked for mypy)
            return None

        Delta_q_max = pin.difference(
            self.model, q, self.model.upperPositionLimit
        )[self.indices]
        Delta_q_min = pin.difference(
            self.model, self.model.lowerPositionLimit, q
        )[self.indices]
        dt_sq = dt * dt
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack(
            [
                np.minimum(
                    self.a_max * dt_sq + self.Delta_q_prev,
                    np.sqrt(2 * self.a_max * Delta_q_max),
                ),
                np.minimum(
                    self.a_max * dt_sq - self.Delta_q_prev,
                    np.sqrt(2 * self.a_max * Delta_q_min),
                ),
            ]
        )
        return G, h
