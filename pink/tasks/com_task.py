#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Stéphane Caron, Simeon Nedelchev, Ivan Domrachev

"""Center of mass task implementation."""

from typing import Optional, Sequence, Union

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from ..exceptions import TargetNotSet, TaskDefinitionError
from .task import Task


class ComTask(Task):
    r"""Regulate the position of the center of mass.

    Attributes:
        target_com: Target position of the center of mass.

    Costs are designed so that errors with varying SI units, here position
    displacements, can be cast to homogeneous values. For example, if task
    "foo" has a position cost of 1.0 and task "bar" a position cost of 0.1,
    then a 1 [cm] error in task "foo" costs as much as a 10 [cm] error in task
    "bar".
    """

    target_com: Optional[np.ndarray]

    def __init__(
        self,
        cost: Union[float, Sequence[float]],
        lm_damping: float = 0.0,
        gain: float = 1.0,
    ) -> None:
        r"""Define a new center of mass task.

        Args:
            cost: Contribution of position errors to the normalized cost, in
                :math:`[\mathrm{cost}] / [\mathrm{m}]`. If this is a vector,
                the cost is anisotropic and each coordinate corresponds to an
                axis of the frame.
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
        self.target_com = None
        self.set_cost(cost)

    def set_cost(
        self, cost: Union[float, Sequence[float], np.ndarray]
    ) -> None:
        r"""Set a new cost for all 3D CoM position coordinates.

        Args:
            cost: Contribution of position errors to the normalized cost, in
                :math:`[\mathrm{cost}] / [\mathrm{m}]`. If this is a vector,
                the cost is anisotropic and each coordinate corresponds to an
                axis of the frame.
        """
        if isinstance(cost, float):
            assert cost >= 0.0
        else:  # not isinstance(cost, float)
            assert all(cost >= 0.0 for cost in cost)
        if isinstance(self.cost, np.ndarray):
            self.cost[0:3] = cost
        else:  # self.cost is not a vector
            raise TaskDefinitionError(
                f"CoM task cost should be a vector, currently cost={self.cost}"
            )

    def set_target(
        self,
        target_com: np.ndarray,
    ) -> None:
        """Set the target CoM position in the world frame.

        Args:
            target_com: Target position of the center of mass in the world
                frame.
        """
        self.target_com = target_com.copy()

    def set_target_from_configuration(
        self, configuration: Configuration
    ) -> None:
        """Set the task target pose from a given robot configuration.

        Args:
            configuration: Robot configuration used to compute the target CoM.
        """
        q = configuration.q
        desired_com = pin.centerOfMass(
            configuration.model, configuration.data, q
        )
        self.set_target(desired_com)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the CoM task error.

        The error is the difference between the target CoM position and the
        current CoM position, both expressed in the world frame.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            CoM task error :math:`e(q)`.
        """
        if self.target_com is None:
            raise TargetNotSet("no target set for CoM")
        q = configuration.q
        actual_com = pin.centerOfMass(
            configuration.model, configuration.data, q
        )
        error = actual_com - self.target_com
        return error

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the CoM task Jacobian.

        The task Jacobian :math:`J(q) \in \mathbb{R}^{3 \times n_v}` is the
        derivative of the CoM position with respect to the configuration
        :math:`q`.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Jacobian matrix :math:`J`.
        """
        if self.target_com is None:
            raise TargetNotSet("no target set for CoM")
        q = configuration.q
        J = pin.jacobianCenterOfMass(
            configuration.model, configuration.data, q
        )
        return J

    def __repr__(self):
        """Human-readable representation of the task."""
        cost = self.cost if isinstance(self.cost, float) else self.cost[0:3]
        return (
            "ComTask("
            f"target_com={self.target_com}, "
            f"cost={cost}, "
            f"gain={self.gain}, "
            f"lm_damping={self.lm_damping})"
        )
