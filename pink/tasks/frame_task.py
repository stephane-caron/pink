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

"""Body task implementation."""

from typing import Optional, Sequence, Union

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from .exceptions import TargetNotSet
from .task import Task
from .utils import body_minus


class FrameTask(Task):
    r"""Regulate the pose of a robot body in the world frame.

    Attributes:
        body: Body frame name, typically the link name from the URDF.
        transform_target_to_world: Target pose for the body frame.

    Costs are designed so that position/orientation costs can be compared
    between tasks. For example, if task 1 has a position cost of 1.0 and task 2
    a position cost of 0.1, then a 1 [cm] error in task 1 costs as much as a 10
    [cm] error in task 2.

    Note:
        Dimensionally, the 6D cost vector is a (normalized) force screw and our
        objective function is a (normalized) energy.
    """

    body: str
    lm_damping: float
    transform_target_to_world: Optional[pin.SE3]

    def __init__(
        self,
        body: str,
        position_cost: Union[float, Sequence[float]],
        orientation_cost: Union[float, Sequence[float]],
        lm_damping: float = 0.0,
    ) -> None:
        r"""Define a new body task.

        Args:
            body: Name of the body frame to move to the target pose.
            position_cost: Contribution of position errors to the normalized
                cost, in :math:`[\mathrm{cost}] / [\mathrm{m}]`. If this is a
                vector, the cost is anisotropic and each coordinate corresponds
                to an axis in the local body frame.
            orientation_cost: Contribution of orientation errors to the
                normalized cost, in :math:`[\mathrm{cost}] / [\mathrm{rad}]`.
                If this is a vector, the cost is anisotropic and each
                coordinate corresponds to an axis in the local body frame.
            lm_damping: Levenberg-Marquardt damping (see class attributes). The
                default value is conservatively low.
        """
        super().__init__(cost=np.ones(6), lm_damping=lm_damping)
        self.body = body
        self.lm_damping = lm_damping
        self.transform_target_to_world = None
        #
        self.set_position_cost(position_cost)
        self.set_orientation_cost(orientation_cost)

    def set_position_cost(
        self, position_cost: Union[float, Sequence[float]]
    ) -> None:
        r"""Set a new cost for all 3D position coordinates.

        Args:
            position_cost: Contribution of position errors to the normalized
                cost, in :math:`[\mathrm{cost}] / [\mathrm{m}]`. If this is a
                vector, the cost is anisotropic and each coordinate corresponds
                to an axis in the local body frame.
        """
        if isinstance(position_cost, float):
            assert position_cost >= 0.0
        else:  # not isinstance(position_cost, float)
            assert all(cost >= 0.0 for cost in position_cost)
        if isinstance(self.cost, float):
            self.cost = self.cost * np.ones(6)
        self.cost[0:3] = position_cost

    def set_orientation_cost(
        self, orientation_cost: Union[float, Sequence[float]]
    ) -> None:
        r"""Set a new cost for all 3D orientation coordinates.

        Args:
            orientation_cost: Contribution of orientation errors to the
                normalized cost, in :math:`[\mathrm{cost}] / [\mathrm{rad}]`.
                If this is a vector, the cost is anisotropic and each
                coordinate corresponds to an axis in the local body frame.
        """
        if isinstance(orientation_cost, float):
            assert orientation_cost >= 0.0
        else:  # not isinstance(orientation_cost, float)
            assert all(cost >= 0.0 for cost in orientation_cost)
        if isinstance(self.cost, float):
            self.cost = self.cost * np.ones(6)
        self.cost[3:6] = orientation_cost

    def set_target(
        self,
        transform_target_to_world: pin.SE3,
    ) -> None:
        """Set task target pose in the world frame.

        Args:
            transform_target_to_world: Transform from the task target frame to
                the world frame.
        """
        self.transform_target_to_world = transform_target_to_world.copy()

    def set_target_from_configuration(
        self, configuration: Configuration
    ) -> None:
        """Set task target pose from a robot configuration.

        Args:
            configuration: Robot configuration.
        """
        self.set_target(configuration.get_transform_frame_to_world(self.body))

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute body task error.

        Mathematically this error is a twist :math:`e(q) \in se(3)` expressed
        in the local frame (i.e., it is a bodytwist ). We map it to
        :math:`\mathbb{R}^6` using Pinocchio's convention (linear coordinates
        followed by angular coordinates).

        The error is the right-minus difference between target and current body
        configuration:

        .. math::

            e(q) := {}_b \xi_{0b} = -(T_{t0} \boxminus T_{b0})
            = -\log(T_{t0} \cdot T_{0b}) = -\log(T_{tb}) = \log(T_{bt})

        where :math:`b` denotes the body frame, :math:`t` the target frame and
        :math:`0` the inertial frame.

        See :func:`Task.compute_error` for more context, and [MLT]_ for details
        on the right-minus operator.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Body task error :math:`e(q)`.
        """
        if self.transform_target_to_world is None:
            raise TargetNotSet(f"no target set for body {self.body}")
        transform_frame_to_world = configuration.get_transform_frame_to_world(
            self.body
        )
        error_in_frame: np.ndarray = body_minus(
            self.transform_target_to_world,
            transform_frame_to_world,
        )
        return error_in_frame

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the body task Jacobian.

        The task Jacobian :math:`J(q) \in \mathbb{R}^{6 \times n_v}` appears in
        the task dynamics:

        .. math::

            J(q) \Delta q = \alpha e(q)

        The derivation of the formula for this Jacobian is detailed in
        [FrameTaskJacobian]_. See also :func:`Task.compute_jacobian` for more
        context on task Jacobians.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Pair :math:`(J, \alpha e)` of Jacobian matrix and error vector,
            both expressed in the body frame.
        """
        jacobian_in_frame = configuration.get_frame_jacobian(self.body)

        # TODO(scaron): fix sign of error and box minus
        if self.transform_target_to_world is None:
            raise TargetNotSet(f"no target set for body {self.body}")
        transform_frame_to_world = configuration.get_transform_frame_to_world(
            self.body
        )
        transform_frame_to_target = (
            self.transform_target_to_world.inverse() * transform_frame_to_world
        )
        J = pin.Jlog6(transform_frame_to_target) @ jacobian_in_frame
        return J

    def __repr__(self):
        """Human-readable representation of the task."""
        orientation_cost = (
            self.cost if isinstance(self.cost, float) else self.cost[3:6]
        )
        position_cost = (
            self.cost if isinstance(self.cost, float) else self.cost[0:3]
        )
        return (
            f"FrameTask({self.body}, "
            f"gain={self.gain}, "
            f"orientation_cost={orientation_cost}, "
            f"position_cost={position_cost}, "
            f"target={self.transform_target_to_world})"
        )
