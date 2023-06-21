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

    def __init__(
        self,
        joint_name_list: Sequence[str],
        ratios: Sequence[float],
        cost: float,
        configuration: Configuration,
        lm_damping: float = 0.0,
    ) -> None:
        r"""Compute Jacobian matrix of a linear holonomic constraint.

        Args:
            joint_name_list: a list of joint names consisting of a linear
                holonomic constraint.
            ratios: a list of ratios that each joint takes in a linear
                holonomic constraint.
            cost: Cost of a linear holonomic task.
            configuration: Robot configuration.
            lm_damping: Unitless scale of the Levenberg-Marquardt (only when
                the error is large) regularization term, which helps when
                targets are unfeasible. Increase this value if the task is too
                jerky under unfeasible targets, but beware that too large a
                damping can slow down the task.
        """
        assert len(joint_name_list) == len(ratios)

        A = np.zeros((1, configuration.model.nv))
        for joint, ratio in zip(joint_name_list, ratios):
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
