#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Stéphane Caron
# Copyright 2023 Inria
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

"""Subset of bounded joints associated with a robot model."""

from typing import List, Optional, Tuple

import numpy as np
import pinocchio as pin

from .limit import Limit


class ConfigurationLimit(Limit):
    """Subspace of the tangent space restricted to joints with position limits.

    Attributes:
        config_limit_gain: gain between 0 and 1 to steer away from
            configuration limits. It is described in "Real-time prioritized
            kinematic control under inequality constraints for redundant
            manipulators" (Kanoun, 2012). More details in `this writeup
            <https://scaron.info/teaching/inverse-kinematics.html>`__.
        model: Robot model the limit applies to.
        joints: Joints with configuration limits.
        projection_matrix: Projection from tangent space to subspace with
            configuration-limited joints.
    """

    config_limit_gain: float
    indices: np.ndarray
    model: pin.Model
    joints: list
    projection_matrix: Optional[np.ndarray]

    def __init__(self, model: pin.Model, config_limit_gain: float = 0.5):
        """Initialize bounded tangent of a model.

        Args:
            model: robot model.
            config_limit_gain: gain between 0 and 1 to steer away from
                configuration limits. See the full class documentation.
        """
        assert 0.0 < config_limit_gain <= 1.0

        has_configuration_limit = np.logical_and(
            model.upperPositionLimit < 1e20,
            model.upperPositionLimit > model.lowerPositionLimit + 1e-10,
        )

        joints = [
            joint
            for joint in model.joints
            if joint.idx_q >= 0
            and has_configuration_limit[
                slice(joint.idx_q, joint.idx_q + joint.nq)
            ].all()
        ]

        index_list: List[int] = []
        for joint in joints:
            index_list.extend(range(joint.idx_v, joint.idx_v + joint.nv))
        indices = np.array(index_list)
        indices.setflags(write=False)

        dim = len(indices)
        projection_matrix = np.eye(model.nv)[indices] if dim > 0 else None

        self.config_limit_gain = config_limit_gain
        self.indices = indices
        self.joints = joints
        self.model = model
        self.projection_matrix = projection_matrix

    def compute_qp_inequalities(
        self,
        q: np.ndarray,
        dt: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        r"""Compute the configuration-dependent velocity limits.

        Those limits are returned as:

        .. math::

            {q \ominus q_{min}} \leq \Delta q \leq {q_{max} \ominus q}

        where :math:`q \in {\cal C}` is the robot's configuration and
        :math:`\Delta q \in T_q({\cal C})` is the displacement in the tangent
        space at :math:`q`. These limits correspond to the derivative of
        :math:`q_{min} \leq q \leq q_{max}`.

        Args:
            q: Robot configuration.
            dt: Integration timestep in [s].

        Returns:
            Pair :math:`(G, h)` representing the inequality constraint as
            :math:`G \Delta q \leq h`, or ``None`` if there is no limit.
        """
        if not self.joints:
            return None

        Delta_q_max = pin.difference(
            self.model, q, self.model.upperPositionLimit
        )
        Delta_q_min = pin.difference(
            self.model, q, self.model.lowerPositionLimit
        )
        p_max = self.config_limit_gain * Delta_q_max[self.indices]
        p_min = self.config_limit_gain * Delta_q_min[self.indices]
        G = np.vstack([self.projection_matrix, -self.projection_matrix])
        h = np.hstack([p_max, -p_min])
        return G, h
