#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 3022 Domrachev Ivan, Simeon Nedelchec

"""Self Collision Avoidance Barrier with hpp-fcl"""

from typing import Union

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from .barrier import Barrier


class SelfCollisionBarrier(Barrier):
    r"""Self Collision Avoidance Barrier.

    This class defines a barrier function based on the smooth convex collision geometry.
    using hpp-fcl through Pinocchio. Note that for non-smooth collision geometries
    behaviour is undefined

    The barrier function is defined as:

    .. math::

        h(q) = \begin{bmatrix} \ldots \\ d(p^1_i, p^2_i) - d_{min}^2 \\ \ldots \end{bmatrix}
            \quad \forall i \in 0 \ldots N

    where :math:`N` is number of collision pairs, :math:`p^k_i` is the :math:`k-th` body in :math:`i`-th collision pair,
    :math:`d(p^1_i, p^2_i)` is the distance between collision bodies in the pair,
    and :math:`d_{min}` is minimal distance between any collision bodies.

    Note:
        The amount of evaluated collision pairs might not be equal to number of all collision pairs.
        If the amount is less, then only the closest collision pairs will be used.

    Attributes:
        d_min: Minimum distance between collision pairs.
    """

    d_min: float

    def __init__(
        self,
        n_collision_pairs: int,
        gain: Union[float, np.ndarray] = 1.0,
        safe_displacement_gain: float = 1.0,
        d_min: float = 0.02,
    ):
        """Initialization of the SelfCollisionBarrier.

        Args:
            n_collision_pairs: Number of collision pairs.
                Note that the number of collision pairs don't have to be equal to
                the total number of collision pairs in the model. If it is less,
                than only the closest collision pairs will be used.
            gain: Barrier gain. Defaults to 1.0.
            safe_displacement_gain: gain for the safe backup displacement.
                cost term. Defaults to 1.0.
            d_min: Minimum distance between any collision pairs.
                Defaults to 0.02.
        """
        super().__init__(
            dim=n_collision_pairs,
            gain=gain,
            safe_displacement_gain=safe_displacement_gain,
        )

        self.d_min = d_min
        self.__q_prev = None

    def compute_barrier(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the value of the barrier function.

        The barrier function is computed as distnace between all collision pairs.

        The barrier function is defined as:

        .. math::

            h(q) = \begin{bmatrix} \ldots \\ d(p^1_i, p^2_i) - d_{min}^2 \\ \ldots \end{bmatrix}
                \quad \forall i \in 0 \ldots N

        where :math:`N` is number of collision pairs, :math:`p^k_i` is the :math:`k-th` body in :math:`i`-th collision pair,
        :math:`d(p^1_i, p^2_i)` is the distance between collision bodies in the pair,
        and :math:`d_{min}` is minimal distance between any collision bodies.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Value of the barrier function
                 :math:`h(q)`.
        """
        return np.array(
            [
                configuration.collision_data.distanceResults[k].min_distance
                - self.d_min
                for k in range(
                    len(configuration.collision_model.collisionPairs)
                )
            ]
        )

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the Jacobian matrix of the barrier function.

        The jacobian matrix could be represented as stacked gradients of each
        collision pair distance function w.r.t. joints. They are computed based on
        frames jacobians and normal surface vector at nearest distance points n.

        The gradient, a.k.a i-th row in the jacobian matrix, is given by:

        .. math::

            J_i = n_1^T J^1_p + (r_1 \times n_1)^T J^1_w + n_2^T J^2_p + (r_2 \times n_2)^T J^2_w,

        where :math:`n_{1,2}` are normal vectors (note that :math:`n_1 = -n_2`), :math:`r_{1, 2}` are vectors from
        frame origin and nearest point, :math:`J^{1, 2}_{p, w}` are position/orientation Jacobians of
        respective frame.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Jacobian matrix
                :math:`\frac{\partial h}{\partial q}(q)`.
        """

        # Get robot and collision models and datas
        model = configuration.model
        data = configuration.data
        collision_model = configuration.collision_model
        collision_data = configuration.collision_data

        J = np.zeros((self.dim, model.nq))

        # Calculate `dim` closest collision pairs, and evaluate them
        N_collision = len(collision_model.collisionPairs)
        distances = np.array(
            [
                collision_data.distanceResults[i].min_distance
                for i in range(N_collision)
            ]
        )
        closest_pairs_idxs = np.argpartition(-distances, -self.dim)[
            -self.dim :
        ]

        for i in range(self.dim):
            # Index of the pair
            k = int(closest_pairs_idxs[i])

            cp = collision_model.collisionPairs[k]
            dr = collision_data.distanceResults[k]

            go_1 = collision_model.geometryObjects[cp.first]
            go_2 = collision_model.geometryObjects[cp.second]

            f1_id = go_1.parentFrame
            f2_id = go_2.parentFrame

            w1 = np.array(dr.getNearestPoint1())
            w2 = np.array(dr.getNearestPoint2())

            # Vectors from frame origin to nearest points
            r1 = np.array(w1 - data.oMf[f1_id].translation)
            r2 = np.array(w2 - data.oMf[f2_id].translation)

            # Normal vector betwee nearest points (n_1 in the notation above)
            n = (w1 - w2) / np.linalg.norm(w1 - w2)

            # Calculate first two terms using first frame jacobian
            J_1 = pin.getFrameJacobian(
                model, data, f1_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            Jrow_v = n.T @ J_1[:3, :] + (pin.skew(r1) @ n).T @ J_1[3:, :]

            # Calculate second two terms using second frame jacobian
            # Note that minus appears, since n_2 = -n_1
            J_2 = pin.getFrameJacobian(
                model, data, f2_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            Jrow_v -= n.T @ J_2[:3, :] + (pin.skew(r2) @ n).T @ J_2[3:, :]

            J[i] = Jrow_v.copy()

        # If collision is undefined, or during the collision, some values
        # might be nans. In this case, set them to zero.
        # Note that for non-colliding smooth convex functions no nans are present.
        J = np.nan_to_num(J)

        return J