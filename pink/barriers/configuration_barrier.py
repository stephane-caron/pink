#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2023 Inria

"""Subset of bounded joints associated with a robot model."""

from typing import List, Optional, Union

import numpy as np
import pinocchio as pin

from pink.configuration import Configuration

from .barrier import CBF


class ConfigurationCBF(CBF):
    """..."""

    indices: np.ndarray
    model: pin.Model
    joints: list
    projection_matrix: Optional[np.ndarray]

    def __init__(
        self,
        model: pin.Model,
        gain: Union[float, np.ndarray] = 0.5,
        r: float = 3.0,
    ):
        """..."""

        has_configuration_limit = np.logical_and(
            model.hasConfigurationLimit(),
            np.logical_and(
                model.upperPositionLimit < 1e20,
                model.upperPositionLimit > model.lowerPositionLimit + 1e-10,
            ),
        )

        joints = [
            joint
            for joint in model.joints
            if joint.idx_q >= 0
            and has_configuration_limit[
                slice(
                    joint.idx_q,
                    joint.idx_q + joint.nq,
                )
            ].all()
        ]

        index_list: List[int] = []
        for joint in joints:
            index_list.extend(range(joint.idx_v, joint.idx_v + joint.nv))
        indices = np.array(index_list)
        indices.setflags(write=False)

        dim = 2 * len(indices)
        projection_matrix = np.eye(model.nv)[indices] if dim > 0 else None

        super().__init__(
            dim,
            gain=gain,
            class_k_fn=lambda h: h / (1 + np.linalg.norm(h)),
            r=r,
        )

        self.indices = indices
        self.joints = joints
        self.model = model
        self.projection_matrix = projection_matrix

    def compute_barrier(self, configuration: Configuration) -> np.ndarray:
        """..."""
        q = configuration.q
        delta_q_max = pin.difference(self.model, q, self.model.upperPositionLimit)
        delta_q_min = pin.difference(self.model, q, self.model.lowerPositionLimit)
        return np.hstack([-delta_q_min[self.indices], delta_q_max[self.indices]])

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """..."""
        return np.vstack([self.projection_matrix, -self.projection_matrix])
