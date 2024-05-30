#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Stéphane Caron, Simeon Nedelchev, Ivan Domrachev

"""Frame task implementation."""

from typing import Optional, Sequence, Union

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from .exceptions import TargetNotSet, TaskDefinitionError
from .task import Task

# TODO: Rewrite doc strings

class ComTask(Task):
    r"""Regulate the pose of a robot frame in the world frame.

    Attributes:
        frame: Frame name, typically the name of a link or joint from the robot
            description.
        transform_target_to_world: Target pose of the frame.

    Costs are designed so that errors with varying SI units, here position and
    orientation displacements, can be cast to homogeneous values. For example,
    if task "foo" has a position cost of 1.0 and task "bar" a position cost of
    0.1, then a 1 [cm] error in task "foo" costs as much as a 10 [cm] error in
    task "bar".

    Note:
        Dimensionally, the 6D cost vector is a (normalized) force screw and our
        objective function is a (normalized) energy.
    """

    frame: str
    target_com: np.ndarray


    def __init__(
        self,
        cost: Union[float, Sequence[float]],
        lm_damping: float = 0.0,
        gain: float = 1.0,
    ) -> None:
        r"""Define a new frame task.

        Args:
            frame: Frame name, typically the name of a link or joint from the
                robot description.
            position_cost: Contribution of position errors to the normalized
                cost, in :math:`[\mathrm{cost}] / [\mathrm{m}]`. If this is a
                vector, the cost is anisotropic and each coordinate corresponds
                to an axis of the frame.
            orientation_cost: Contribution of orientation errors to the
                normalized cost, in :math:`[\mathrm{cost}] / [\mathrm{rad}]`.
                If this is a vector, the cost is anisotropic and each
                coordinate corresponds to an axis of the frame.
            lm_damping: Levenberg-Marquardt damping (see class attributes). The
                default value is conservatively low.
            gain: Task gain :math:`\alpha \in [0, 1]` for additional low-pass
                filtering. Defaults to 1.0 (no filtering) for dead-beat
                control.
        """
        super().__init__(
            cost=np.ones(3),  # updated below
            gain=gain,
            lm_damping=lm_damping,
        )

        self.lm_damping = lm_damping
        self.target_com = None
        #
        self.set_cost(cost)

    def set_cost(
        self, cost: Union[float, Sequence[float], np.ndarray]
    ) -> None:
        r"""Set a new cost for all 3D CoM position coordinates.

        Args:
            position_cost: Contribution of position errors to the normalized
                cost, in :math:`[\mathrm{cost}] / [\mathrm{m}]`. If this is a
                vector, the cost is anisotropic and each coordinate.
        """
        if isinstance(cost, float):
            assert cost >= 0.0
        else:  # not isinstance(position_cost, float)
            assert all(cost >= 0.0 for cost in cost)
        if isinstance(self.cost, np.ndarray):
            self.cost[0:3] = cost
        else:  # self.cost is not a vector
            raise TaskDefinitionError(
                "CoM task cost should be a vector, "
                f"currently cost={self.cost}"
            )


    def set_target(
        self,
        target_com: np.ndarray,
    ) -> None:
        """Set task target pose in the world frame.

        Args:
            transform_target_to_world: Transform from the task target frame to
                the world frame.
        """
        self.target_com = target_com.copy()

    def set_target_from_configuration(
        self, configuration: Configuration
    ) -> None:
        """Set task target pose from a robot configuration.

        Args:
            configuration: Robot configuration.
        """
        q = configuration.q.copy()
        desired_com = pin.centerOfMass(configuration.model, configuration.data, q)
        self.set_target(desired_com)


    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute frame task error.

        This error is a twist :math:`e(q) \in se(3)` expressed in the local
        frame (i.e. it is a *body* twist). We map it to :math:`\mathbb{R}^6`
        using Pinocchio's convention where linear coordinates are followed by
        angular coordinates.

        The error is the right-minus difference between the target pose
        :math:`T_{0t}` and current frame pose :math:`T_{0b}`:

        .. math::

            e(q) := {}_b \xi_{0b} = -(T_{t0} \boxminus T_{b0})
            = -\log(T_{t0} \cdot T_{0b}) = -\log(T_{tb}) = \log(T_{bt})

        where :math:`b` denotes our frame, :math:`t` the target frame and
        :math:`0` the inertial frame.

        See :func:`pink.tasks.task.Task.compute_error` for more context, and
        [MLT]_ for details on the right-minus operator.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Frame task error :math:`e(q)`.
        """
        if self.target_com is None:
            raise TargetNotSet(f"no target set for CoM")
        
        q = configuration.q.copy()
        error = pin.centerOfMass(configuration.model, configuration.data, q) - self.target_com
        return error

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the frame task Jacobian.

        The task Jacobian :math:`J(q) \in \mathbb{R}^{6 \times n_v}` is the
        derivative of the task error :math:`e(q) \in \mathbb{R}^6` with respect
        to the configuration :math:`q`. The formula for the frame task is:

        .. math::

            J(q) = -\text{Jlog}_6(T_{tb}) {}_b J_{0b}(q)

        The derivation of the formula for this Jacobian is detailed in
        [FrameTaskJacobian]_. See also
        :func:`pink.tasks.task.Task.compute_jacobian` for more context on task
        Jacobians.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Jacobian matrix :math:`J`, expressed locally in the frame.
        """
        if self.target_com is None:
            raise TargetNotSet(f"no target set for CoM")
        q = configuration.q.copy()
        J = pin.jacobianCenterOfMass(configuration.model, configuration.data, q)
        return J

    def __repr__(self):
        """Human-readable representation of the task."""
        position_cost = (
            self.cost if isinstance(self.cost, float) else self.cost[0:3]
        )
        return (
            "ComTask("
            f"gain={self.gain}, "
            f"com_cost={position_cost}"
        )
