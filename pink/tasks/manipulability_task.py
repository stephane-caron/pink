#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""A task to maximize the manipulability of a robot manipulator."""

from typing import Literal, override

import numpy as np
import pinocchio as pin

from ..configuration import Configuration
from .task import Task


class ManipulabilityTask(Task):
    r"""Task to maximize the manipulability of a robot manipulator.

    Manipulability is a scalar measure of how well-conditioned the kinematic
    Jacobian is at the current configuration. Higher manipulability means the
    robot can move more easily in all directions. This task maximizes the
    Yoshikawa manipulability index:

    .. math::

        m(q) = \sqrt{\det(J(q) J(q)^T)}

    We try to minimize:

    .. math::

         || J_m \dot{q} - v_\text{man} || ^2

    This will try to increase the manipulability at a rate
    of :math:`v_\text{man}`.

    Attributes:
        frame: Frame name for which to compute manipulability.
        cost: Weight for the manipulability task in the QP objective.
        reference_frame: Pinocchio reference frame for Jacobian computation.
        manipulability_rate: Desired rate of change of manipulability
        to achieve.

    Note:
        Check the manipulability task of PlaCo for a similar (yet different)
        implementation: https://placo.readthedocs.io/en/latest/kinematics/regularization.html#manipulability-regularization
        This term gives a behavior similar to the MMC controller
        (Manipulability Motion Control) by Jese Haviland and Peter Corke.
        Consider also citing their work if you use this task in a publication: https://jhavl.github.io/mmc/
    """

    def __init__(
        self,
        frame: str,
        cost: float = 1.0,
        lm_damping: float = 0.0,
        gain: float = 1.0,
        reference_frame: pin.ReferenceFrame = pin.ReferenceFrame.LOCAL,
        manipulability_rate: float = 0.1,
        mask: Literal["position", "orientation", "planar_xy"]
        | np.ndarray
        | None = None,
    ) -> None:
        r"""Initialize the manipulability task.

        Args:
            frame: Frame name (from the robot model) for which to compute and
                maximize manipulability. This is typically the end-effector
                frame name.
            cost: Weight for the manipulability task in the QP objective.
                Scales both the Hessian and linear terms. Higher values
                prioritize manipulability over other tasks. Defaults to 1.0.
            lm_damping: Levenberg-Marquardt damping (see class attributes). The
                default value is conservatively low.
            gain: Task gain :math:`\alpha \in [0, 1]` for additional low-pass
                filtering. Defaults to 1.0 (no filtering) for dead-beat
                control.
            reference_frame: Pinocchio reference frame for Jacobian
                computation.
                Only WORLD is supported for now. Defaults to WORLD.
            manipulability_rate: Desired rate of change of manipulability to
                achieve, in units of manipulability per second. This is the
                target velocity for the manipulability gradient ascent. Higher
                values will try to increase manipulability more aggressively.
            mask: Predefined mask string or custom binary mask to select which
                components of the Jacobian to consider for manipulability
                computation. The mask should be a binary array of shape (6,)
                where 1 indicates the corresponding component is included and 0
                means it is ignored. Predefined strings are:
                - "position": Only consider the linear velocity part (first 3
                  components).
                - "orientation": Only consider the angular velocity part (last
                  3 components).
                - "planar_xy": Only consider the x and y linear velocity
                  components (first 2 components). This is useful for
                  planar manipulators.
        """
        super().__init__(
            cost=cost,
            lm_damping=lm_damping,
            gain=gain,
        )
        self.frame = frame
        if reference_frame not in (
            pin.ReferenceFrame.WORLD,
            pin.ReferenceFrame.LOCAL,
        ):
            raise ValueError(
                f"invalid reference frame {reference_frame} for Jacobian "
                "computation; only LOCAL and WORLD are supported"
            )
        self._cost_float = cost
        self.reference_frame = reference_frame
        self.manipulability_rate = manipulability_rate
        self.mask = self._get_validated_mask(mask)

    def _get_validated_mask(
        self,
        mask: Literal["position", "orientation", "planar_xy"]
        | np.ndarray
        | None,
    ) -> np.ndarray | None:
        if mask is None:
            return None

        if isinstance(mask, np.ndarray):
            if mask.shape != (6,):
                raise ValueError(
                    f"custom mask must have shape (6,), got {mask.shape}"
                )
            if not np.all(np.isin(mask, [0, 1])):
                raise ValueError("custom mask must be binary (0 or 1)")
            return mask

        elif isinstance(mask, str):
            return self._get_mask_from_string(mask)

        else:
            raise ValueError(
                "mask must be either a predefined string or"
                "a custom binary numpy array"
            )

    def _get_mask_from_string(
        self,
        mask_str: Literal["position", "orientation", "planar_xy"],
    ) -> np.ndarray | None:
        if mask_str == "position":
            return np.array([1, 1, 1, 0, 0, 0])
        elif mask_str == "orientation":
            return np.array([0, 0, 0, 1, 1, 1])
        elif mask_str == "planar_xy":
            return np.array([1, 1, 0, 0, 0, 0])
        else:
            raise ValueError(f"invalid mask string: {mask_str}")

    def _mask_jacobian(self, J: np.ndarray) -> np.ndarray:
        """Apply the mask to the Jacobian matrix."""
        if self.mask is None:
            return J
        mask_expanded = self.mask[:, np.newaxis].repeat(J.shape[1], axis=1)
        return J[mask_expanded.astype(bool)].reshape(
            int(self.mask.sum()), J.shape[1]
        )

    def _mask_hessian(self, H: np.ndarray) -> np.ndarray:
        """Apply the mask to the kinematic Hessian tensor."""
        if self.mask is None:
            return H
        mask_expanded = (
            self.mask[np.newaxis, :, np.newaxis]
            .repeat(H.shape[0], axis=0)
            .repeat(H.shape[2], axis=2)
        )
        return H[mask_expanded.astype(bool)].reshape(
            H.shape[0], int(self.mask.sum()), H.shape[2]
        )

    def compute_manipulability(self, configuration: Configuration) -> float:
        r"""Compute the Yoshikawa manipulability index.

        The manipulability index is defined as:

        .. math::

            w(q) = \det(J(q) J(q)^T)

        where :math:`J(q)` is the 6xn Jacobian matrix of the frame. This is
        the square of the Yoshikawa manipulability measure. A higher value
        indicates better conditioning of the Jacobian, meaning the robot can
        generate velocities more uniformly in all Cartesian directions.

        Args:
            configuration: Robot configuration at which to evaluate
                manipulability.

        Returns:
            The manipulability index :math:`\det(J J^T)`.

        Note:
            When the Jacobian becomes singular (at kinematic singularities),
            the manipulability approaches zero.
        """
        frame_id = configuration.model.getFrameId(self.frame)
        J0 = pin.getFrameJacobian(
            configuration.model,
            configuration.data,
            frame_id,
            self.reference_frame,
        )
        J0_masked = self._mask_jacobian(J0)
        return float(np.linalg.det(J0_masked @ J0_masked.T))

    def compute_kinematic_hessian(
        self, configuration: Configuration
    ) -> np.ndarray:
        r"""Compute the kinematic Hessian tensor.

        The kinematic Hessian :math:`H \in \mathbb{R}^{n \times 6 \times n}` is
        the derivative of the Jacobian with respect to joint positions. For
        each pair of joints (i, j), it captures how the i-th column of the
        Jacobian changes when joint j moves.

        The Hessian is computed using the relation:

        .. math::

            H_{ijk} = \frac{\partial J_{ki}}{\partial q_j}

        The Hessian can be used to compute the time derivative of the Jacobian:

        .. math::
            \dot{J} = H \cdot \dot{q}

        For more details consider reading "Manipulator Differential Kinematics
        Part 2: Acceleration and Advanced Applications" by Jesse Haviland and
        Peter Corke.

        Args:
            configuration: Robot configuration at which to evaluate the
                kinematic Hessian.

        Returns:
            The kinematic Hessian tensor of shape (n, 6, n), where n is the
            number of degrees of freedom (model.nv).

        Note:
            This implementation assumes all joints are revolute.
        """
        frame_id = configuration.model.getFrameId(self.frame)

        J0 = pin.getFrameJacobian(
            configuration.model,
            configuration.data,
            frame_id,
            self.reference_frame,
        )
        n = configuration.model.nv

        H = np.zeros((n, 6, n))

        for j in range(n):
            for i in range(j, n):
                # Linear velocity part: omega_j x v_i
                H[j, :3, i] = np.cross(J0[3:, j], J0[:3, i])
                # Angular velocity part: omega_j x omega_i
                H[j, 3:, i] = np.cross(J0[3:, j], J0[3:, i])

                if i != j:
                    # Symmetry for linear part
                    H[i, :3, j] = H[j, :3, i]

        H_masked = self._mask_hessian(H)
        return H_masked

    @override
    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the manipulability gradient with respect to joint angles.

        The manipulability Jacobian :math:`J_m \in \mathbb{R}^{n}` is the
        gradient of the manipulability with respect to joint positions:

        .. math::
            \dot{m} = J_m^\top \dot{q} \quad \text{where} \quad
            J_m^\top = \frac{\partial m}{\partial q} = m \cdot
            \begin{pmatrix}
            (\text{vec}(J H_1) \cdot \text{vec}\left((J J^\top)^{-1}\right) \\
            (\text{vec}(J H_2) \cdot \text{vec}\left((J J^\top)^{-1}\right) \\
            \vdots \\
            (\text{vec}(J H_n) \cdot \text{vec}\left((J J^\top)^{-1}\right) \\
            \end{pmatrix} \in \mathbb{R}^{n}

        This gradient is used to formulate a gradient ascent direction for
        maximizing manipulability.

        Args:
            configuration: Robot configuration at which to evaluate the
                manipulability gradient.

        Returns:
            The manipulability Jacobian as a row vector of shape (1, n),
            where n is the number of degrees of freedom which
            depends on the masking.
        """
        frame_id = configuration.model.getFrameId(self.frame)
        J = pin.getFrameJacobian(
            configuration.model,
            configuration.data,
            frame_id,
            self.reference_frame,
        )
        J_masked = self._mask_jacobian(J)
        H = self.compute_kinematic_hessian(configuration)
        manipulability = self.compute_manipulability(configuration)

        n = J_masked.shape[1]

        JJTinv = np.linalg.pinv(J_masked @ J_masked.T)
        Jm = np.zeros((n, 1))

        for i in range(n):
            JH = J_masked @ H[i, :, :].T
            Jm[i, 0] = manipulability * JH.flatten("F").T @ JJTinv.flatten("F")
        return Jm.T

    @override
    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute task error.

        Here the error is the desired rate of change of manipulability,
        which we want to achieve by following the gradient ascent direction.
        The error is defined as:

        .. math::

            e(q) = - v_\text{man} \in \mathbb{R}

        If the `manipulability_rate` is positive, this will try to increase
        the manipulability.
        Choosing a negative value would try to decrease it, which can be used
        to find singular configurations.

        Args:
            configuration: Robot configuration.

        Returns:
            Task error vector.
        """
        return np.array([-self.manipulability_rate])

    @override
    def __repr__(self) -> str:
        return (
            f"ManipulabilityTask(frame={self.frame}, "
            f"cost={self.cost}, "
            f"lm_damping={self.lm_damping}, "
            f"gain={self.gain}, "
            f"manipulability_rate={self.manipulability_rate})"
        )
