#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2023 Inria

"""Subset of velocity-limited joints in a robot model."""

from typing import List, Optional, Tuple

import numpy as np
import pinocchio as pin

from .limit import Limit


class VelocityLimit(Limit):
    """Subset of velocity-limited joints in a robot model.

    Attributes:
        indices: Tangent indices corresponding to velocity-limited joints.
        joints: List of velocity-limited joints.
        model: Robot model.
        projection_matrix: Projection from tangent space to subspace with
            velocity-limited joints.
    """

    indices: np.ndarray
    joints: list
    model: pin.Model
    projection_matrix: Optional[np.ndarray]

    def __init__(self, model: pin.Model):
        """Initialize bounded tangent of a model.

        Args:
            model: robot model.
        """
        has_velocity_limit = np.logical_and(
            model.velocityLimit < 1e20,
            model.velocityLimit > 1e-10,
        )

        joints = [
            joint
            for joint in model.joints
            if joint.idx_v >= 0
            and has_velocity_limit[
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

        self.indices = indices
        self.joints = joints
        self.model = model
        self.projection_matrix = projection_matrix

    def compute_qp_inequalities(
        self,
        q: np.ndarray,
        dt: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        r"""Compute inequalities for velocity limits.

        Those limits are defined by:

        .. math::

            -\mathrm{d}t v_{max} \leq \Delta q \leq \mathrm{d}t v_{max}

        where :math:`v_{max} \in {\cal T}` is the robot's velocity limit
        vector and :math:`\Delta q \in T_q({\cal C})` is the displacement
        computed by the inverse kinematics.

        Args:
            q: Robot configuration.
            dt: Integration timestep in [s].

        Returns:
            Pair :math:`(G, h)` representing the inequality constraint as
            :math:`G \Delta q \leq h`, or ``None`` if there is no limit.
        """
        if self.projection_matrix is None:  # no joint (thus checked for mypy)
            return None

        v_max = self.model.velocityLimit[self.indices]
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([dt * v_max, dt * v_max])
        return G, h
