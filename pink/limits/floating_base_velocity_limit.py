#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
"""Floating-base velocity limits."""

from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import numpy as np
import pinocchio as pin

from .limit import Limit

if TYPE_CHECKING:
    from ..configuration import Configuration


def _as_velocity_vector(
    values: Union[Sequence[float], float], label: str
) -> np.ndarray:
    """Expand scalar or 3 vector bounds to a NumPy array."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.repeat(array, 3)
    array = array.flatten()
    if array.shape != (3,):
        raise ValueError(
            f"{label} must be a scalar or an iterable of length 3, "
            f"got shape {array.shape}"
        )
    return array


def _find_base_frame(
    model: pin.Model, base_frame: Optional[str]
) -> Tuple[str, int]:
    """Return the name and id of the frame attached to the root joint.

    The frame is either provided explicitly or discovered as the first frame
    attached to ``root_joint``. Raises ``ValueError`` if the requested frame
    does not exist or if no frame is attached to the root joint.
    """
    if base_frame is not None:
        if not model.existFrame(base_frame):
            raise ValueError(
                f"Frame '{base_frame}' does not exist in the model."
            )
        frame_id = model.getFrameId(base_frame)
        return base_frame, frame_id

    root_joint_id = model.getJointId("root_joint")
    for frame in model.frames:
        if frame.parentJoint == root_joint_id:
            return frame.name, model.getFrameId(frame.name)

    raise ValueError("Model does not expose a frame attached to 'root_joint'.")


class FloatingBaseVelocityLimit(Limit):
    """Velocity limits applied to the robot floating base."""

    base_frame: str
    frame_id: int
    linear_max: np.ndarray
    angular_max: np.ndarray
    twist_max: np.ndarray
    model: pin.Model
    root_joint_id: int
    root_idx_v: int
    root_nv: int

    def __init__(
        self,
        model: pin.Model,
        base_frame: Optional[str],
        max_linear_velocity: Union[Sequence[float], float],
        max_angular_velocity: Union[Sequence[float], float],
    ):
        """Store floating-base bounds and Pinocchio handles.

        Args:
            model: Robot model with a floating base joint.
            base_frame: Optional frame attached to the floating base. The
                corresponding Jacobian rows constrain the base twist. Pass
                ``None`` to select the first frame attached to ``root_joint``.
            max_linear_velocity: Linear velocity limits along the base frame
                axes. A scalar applies the same bound to all axes.
            max_angular_velocity: Angular velocity limits around the base frame
                axes. A scalar applies the same bound to all axes.
        """
        self.model = model
        self.linear_max = _as_velocity_vector(
            max_linear_velocity, "max_linear_velocity"
        )
        self.angular_max = _as_velocity_vector(
            max_angular_velocity, "max_angular_velocity"
        )
        self.twist_max = np.hstack([self.linear_max, self.angular_max])
        if not model.existJointName("root_joint"):
            raise ValueError(
                "FloatingBaseVelocityLimit requires a floating-base root "
                "joint."
            )

        self.root_joint_id = model.getJointId("root_joint")
        root_joint = model.joints[self.root_joint_id]
        self.root_idx_v = root_joint.idx_v
        self.root_nv = root_joint.nv

        self.base_frame, self.frame_id = _find_base_frame(model, base_frame)

        parent_joint = model.frames[self.frame_id].parentJoint
        if parent_joint != self.root_joint_id:
            raise ValueError(
                f"Frame '{self.base_frame}' is not attached to the root joint."
            )

    def compute_qp_inequalities(
        self,
        configuration: "Configuration",
        dt: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Linearize floating base velocity bounds."""
        finite_mask = np.isfinite(self.twist_max)
        if not finite_mask.any():
            return None

        jacobian = pin.getFrameJacobian(
            self.model,
            configuration.data,
            self.frame_id,
            pin.ReferenceFrame.LOCAL,
        )

        # Keep only columns belonging to the root joint twist.
        if self.root_idx_v is not None:
            jacobian[:, : self.root_idx_v] = 0.0
            after_start = self.root_idx_v + self.root_nv
            if after_start < jacobian.shape[1]:
                jacobian[:, after_start:] = 0.0

        active_rows = jacobian[finite_mask, :]
        G = np.vstack([active_rows, -active_rows])

        bounds = dt * self.twist_max[finite_mask]
        h = np.hstack([bounds, bounds])

        return G, h
