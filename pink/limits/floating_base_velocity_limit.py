#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
"""Floating-base velocity limits."""
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pinocchio as pin

from .limit import Limit


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


class FloatingBaseVelocityLimit(Limit):
    """Velocity limits applied to the robot floating base."""

    base_frame: str
    frame_id: Optional[int]
    linear_max: np.ndarray
    angular_max: np.ndarray
    model: pin.Model
    root_joint_id: Optional[int]
    root_idx_v: Optional[int]
    root_nv: int
    data: pin.Data

    def __init__(
        self,
        model: pin.Model,
        base_frame: str,
        max_linear_velocity: Union[Sequence[float], float],
        max_angular_velocity: Union[Sequence[float], float],
    ):
        """Store floating-base bounds and Pinocchio handles.

        Args:
            model: Robot model with a floating base joint.
            base_frame: Frame attached to the floating base. The corresponding
            Jacobian rows constrain the base twist.
            max_linear_velocity: Linear velocity limits along the base frame
                axes. A scalar applies the same bound to all axes.
            max_angular_velocity: Angular velocity limits around the base frame
                axes. A scalar applies the same bound to all axes.
        """
        self.model = model
        self.base_frame = base_frame
        self.linear_max = _as_velocity_vector(
            max_linear_velocity, "max_linear_velocity"
        )
        self.angular_max = _as_velocity_vector(
            max_angular_velocity, "max_angular_velocity"
        )
        self.data = model.createData()

        if model.existJointName("root_joint"):
            self.root_joint_id = model.getJointId("root_joint")
            root_joint = model.joints[self.root_joint_id]
            self.root_idx_v = root_joint.idx_v
            self.root_nv = root_joint.nv
        else:
            self.root_joint_id = None
            self.root_idx_v = None
            self.root_nv = 0

        if model.existFrame(base_frame):
            self.frame_id = model.getFrameId(base_frame)
            parent_joint = model.frames[self.frame_id].parentJoint
        else:
            self.frame_id = None
            parent_joint = None

        if self.root_joint_id is not None and self.frame_id is None:
            raise ValueError(
                f"Frame '{base_frame}' does not exist in the model."
            )

        if (
            self.frame_id is not None
            and self.root_joint_id is not None
            and parent_joint != self.root_joint_id
        ):
            raise ValueError(
                f"Frame '{base_frame}' is not attached to the root joint."
            )

    def compute_qp_inequalities(
        self,
        q: np.ndarray,
        dt: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Linearize floating base velocity bounds."""
        if (
            self.frame_id is None
            or self.root_joint_id is None
            or not np.isfinite(self.linear_max).any()
            and not np.isfinite(self.angular_max).any()
        ):
            return None

        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        jacobian = pin.getFrameJacobian(
            self.model,
            self.data,
            self.frame_id,
            pin.ReferenceFrame.LOCAL,
        )

        rows = []
        bounds = []

        for axis, bound in enumerate(self.linear_max):
            if not np.isfinite(bound):
                continue
            row = jacobian[axis, :].copy()
            if self.root_idx_v is not None:
                row[: self.root_idx_v] = 0.0
                after_start = self.root_idx_v + self.root_nv
                if after_start < row.shape[0]:
                    row[after_start:] = 0.0
            rows.extend([row, -row])
            bounds.extend([dt * bound, dt * bound])

        for axis, bound in enumerate(self.angular_max):
            if not np.isfinite(bound):
                continue
            row = jacobian[3 + axis, :].copy()
            if self.root_idx_v is not None:
                row[: self.root_idx_v] = 0.0
                after_start = self.root_idx_v + self.root_nv
                if after_start < row.shape[0]:
                    row[after_start:] = 0.0
            rows.extend([row, -row])
            bounds.extend([dt * bound, dt * bound])

        if not rows:
            return None

        G = np.vstack(rows)
        h = np.asarray(bounds)

        return G, h
