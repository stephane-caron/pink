#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron
# Copyright 2024 Inria

"""Relative frame task implementation."""

from typing import Optional, Sequence, Union

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from .exceptions import TargetNotSet, TaskDefinitionError
from .task import Task


class RelativeFrameTask(Task):
    r"""Regulate the pose of a robot frame relative to another robot frame.

    Attributes:
        frame: Frame name, typically the name of a link or joint from the robot
            description.
        root: Name of the robot frame the task is relative to.
        transform_target_to_root: Target pose of the frame.
    """

    frame: str
    root: str
    transform_target_to_root: Optional[pin.SE3]

    def __init__(
        self,
        frame: str,
        root: str,
        position_cost: Union[float, Sequence[float]],
        orientation_cost: Union[float, Sequence[float]],
        lm_damping: float = 0.0,
        gain: float = 1.0,
    ) -> None:
        r"""Define a relative frame task.

        Args:
            frame: Frame name, typically the name of a link or joint from the
                robot description.
            root: Name of the robot frame the task is relative to.
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
            cost=np.ones(6),  # updated below
            gain=gain,
            lm_damping=lm_damping,
        )
        self.frame = frame
        self.root = root
        self.lm_damping = lm_damping
        self.transform_target_to_root = None
        #
        self.set_position_cost(position_cost)
        self.set_orientation_cost(orientation_cost)

    def set_position_cost(
        self, position_cost: Union[float, Sequence[float], np.ndarray]
    ) -> None:
        r"""Set a new cost for all 3D position coordinates.

        Args:
            position_cost: Contribution of position errors to the normalized
                cost, in :math:`[\mathrm{cost}] / [\mathrm{m}]`. If this is a
                vector, the cost is anisotropic and each coordinate corresponds
                to an axis of the frame.
        """
        if isinstance(position_cost, float):
            assert position_cost >= 0.0
        else:  # not isinstance(position_cost, float)
            assert all(cost >= 0.0 for cost in position_cost)
        if isinstance(self.cost, np.ndarray):
            self.cost[0:3] = position_cost
        else:  # self.cost is not a vector
            raise TaskDefinitionError(
                "Frame task cost should be a vector, "
                f"currently cost={self.cost}"
            )

    def set_orientation_cost(
        self, orientation_cost: Union[float, Sequence[float], np.ndarray]
    ) -> None:
        r"""Set a new cost for all 3D orientation coordinates.

        Args:
            orientation_cost: Contribution of orientation errors to the
                normalized cost, in :math:`[\mathrm{cost}] / [\mathrm{rad}]`.
                If this is a vector, the cost is anisotropic and each
                coordinate corresponds to an axis of the frame.
        """
        if isinstance(orientation_cost, float):
            assert orientation_cost >= 0.0
        else:  # not isinstance(orientation_cost, float)
            assert all(cost >= 0.0 for cost in orientation_cost)
        if isinstance(self.cost, np.ndarray):
            self.cost[3:6] = orientation_cost
        else:  # self.cost is not a vector
            raise TaskDefinitionError(
                "Frame task cost should be a vector, "
                f"currently cost={self.cost}"
            )

    def set_target(
        self,
        transform_target_to_root: pin.SE3,
    ) -> None:
        """Set task target pose in the root frame.

        Args:
            transform_target_to_root: Affine transform matrix (4×4) from the
                task target frame to the root frame.
        """
        self.transform_target_to_root = transform_target_to_root.copy()

    def set_target_from_configuration(
        self, configuration: Configuration
    ) -> None:
        """Set task target pose from a robot configuration.

        Args:
            configuration: Robot configuration.
        """
        self.set_target(configuration.get_transform(self.frame, self.root))

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the error twist.

        The error is a twist :math:`e(q) \in se(3)` expressed in the local
        frame (i.e., it is a *body* twist). We map it to :math:`\mathbb{R}^6`
        following Pinocchio's convention where linear coordinates are followed
        by angular coordinates. The error vector is:

        .. math::

            e(q) := {}_b \xi_{0b} = \log(T_{bt})

        where :math:`b` denotes our frame, :math:`t` the target frame and
        :math:`0` the inertial frame. See also :func:`Task.compute_error` for
        more context.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Frame task error :math:`e(q)`.
        """
        if self.transform_target_to_root is None:
            raise TargetNotSet(
                f"target pose of frame '{self.frame}' "
                f"in frame '{self.root}' is undefined"
            )
        transform_frame_to_root = configuration.get_transform(
            self.frame, self.root
        )
        transform_target_to_frame = transform_frame_to_root.actInv(
            self.transform_target_to_root
        )
        error_in_frame: np.ndarray = pin.log(transform_target_to_frame).vector
        return error_in_frame

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the frame task Jacobian.

        The task Jacobian :math:`J(q) \in \mathbb{R}^{6 \times n_v}` is the
        derivative of the task error :math:`e(q) \in \mathbb{R}^6` with respect
        to the configuration :math:`q`. The formula for the frame task is:

        .. math::

            J(q) = \text{Jlog}_6(T_{bt}) (\text{Ad}_{T_{tr}) {}_r J_{0r}(q) -
            \text{Ad}_{T_{tb}} {}_b J_{0b}(q))

        The formula implemented here is more general than the one detailed in
        [FrameTaskJacobian]_. See also :func:`Task.compute_jacobian` for more
        context on task Jacobians.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Jacobian matrix :math:`J`, expressed locally in the frame.
        """
        if self.transform_target_to_root is None:
            raise TargetNotSet(
                f"target pose of frame '{self.frame}' "
                f"in frame '{self.root}' is undefined"
            )
        transform_root_to_frame = configuration.get_transform(
            self.root, self.frame
        )
        transform_target_to_frame = transform_root_to_frame.act(
            self.transform_target_to_root
        )
        jacobian_frame_in_frame = configuration.get_frame_jacobian(self.frame)
        jacobian_root_in_root = configuration.get_frame_jacobian(self.root)
        action_frame_to_target = transform_target_to_frame.actionInverse
        action_root_to_target = transform_target_to_frame.actInv(
            transform_root_to_frame
        ).action
        J = pin.Jlog6(transform_target_to_frame) @ (
            action_root_to_target @ jacobian_root_in_root
            - action_frame_to_target @ jacobian_frame_in_frame
        )
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
            "RelativeFrameTask("
            f"frame={self.frame}, "
            f"root={self.root}, "
            f"gain={self.gain}, "
            f"orientation_cost={orientation_cost}, "
            f"position_cost={position_cost}, "
            f"transform_target_to_root={self.transform_target_to_root})"
        )
