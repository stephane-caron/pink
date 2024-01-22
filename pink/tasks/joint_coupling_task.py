#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Joint coupling task implementation."""

from typing import Sequence

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from .linear_holonomic_task import LinearHolonomicTask


class JointCouplingTask(LinearHolonomicTask):
    r"""Coupling between two revolute joints.

    Note:
        This task only considers a 1-Dimensional task.
    """

    joint_names: Sequence[str]

    def __init__(
        self,
        joint_names: Sequence[str],
        ratios: Sequence[float],
        cost: float,
        configuration: Configuration,
        lm_damping: float = 0.0,
    ) -> None:
        r"""Compute Jacobian matrix of a linear holonomic constraint.

        Args:
            joint_names: a list of joint names consisting of a linear
                holonomic constraint.
            ratios: a list of ratios that each joint takes in a linear
                holonomic constraint.
            cost: value used to cast joint angle differences to a cost, in
                :math:`[\mathrm{cost}] / [\mathrm{rad}]`.
            configuration: Configuration used to retrieve joint indices from
                joint names.
            lm_damping: Unitless scale of the Levenberg-Marquardt (only when
                the error is large) regularization term, which helps when
                targets are unfeasible. Increase this value if the task is too
                jerky under unfeasible targets, but beware that too large a
                damping can slow down the task.
        """
        assert len(joint_names) == len(ratios)

        A = np.zeros((1, configuration.model.nv))
        for joint, ratio in zip(joint_names, ratios):
            joint_obj = configuration.model.joints[
                configuration.model.getJointId(joint)
            ]
            strt = joint_obj.idx_v
            end = strt + joint_obj.nv
            A[:, strt:end] = ratio

        super().__init__(
            A,
            np.zeros(1),
            pin.neutral(configuration.model),
            cost=cost,
            lm_damping=lm_damping,
        )
        self.joint_names = joint_names
        self.ratios = ratios

    def __repr__(self):
        """Human-readable representation of the task."""
        return (
            "JointCouplingTask("
            f"joint_names={self.joint_names}, "
            f"ratios={self.ratios}, "
            f"cost={self.cost}, "
            f"gain={self.gain}, "
            f"lm_damping={self.lm_damping})"
        )
