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

         || J_m \dot{q} - \dot{m}_\text{desired} || ^2

    This will try to increase the manipulability at a rate
    of :math:`\dot{m}_\text{desired}`.

    Attributes:
        frame: Frame name for which to compute manipulability.
        cost: Weight for the manipulability task in the QP objective.
        reference_frame: Pinocchio reference frame for Jacobian computation.
        manipulability_rate: Desired rate of change of manipulability :math:`\dot{m}_\text{desired}`
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
                computation. LOCAL and WORLD are supported.
                Defaults to LOCAL.
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

            m(q) = \sqrt{\det(J(q) J(q)^T)}

        where :math:`J(q)` is the :math:`6\times n` Jacobian matrix of the frame. This is
        the Yoshikawa manipulability measure. A higher value
        indicates better conditioning of the Jacobian, meaning the robot can
        generate velocities more uniformly in all Cartesian directions.

        Args:
            configuration: Robot configuration at which to evaluate
                manipulability.

        Returns:
            The manipulability index.

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
        return np.sqrt(float(np.linalg.det(J0_masked @ J0_masked.T)))

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

        The computation can be done by dividing the Jacobian into linear and angular parts:

        - For the linear part (first 3 rows), the Hessian is given by:

        .. math::
            H_{j, :3, i} = \omega_j \times v_i

        - For the angular part (last 3 rows), the Hessian is given by:

        .. math::
            H_{j, 3:, i} = \omega_j \times \omega_i

        ...where :math:`\omega_j` is the j-th column of the angular part of the Jacobian
        and :math:`v_i` is the i-th column of the linear part of the Jacobian.

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

        v = J0[:3, :]  # shape (3, n)
        omega = J0[3:, :]  # shape (3, n)

        omega_j = omega.T[:, np.newaxis, :]  # (n, 1, 3)
        v_i = v.T[np.newaxis, :, :]  # (1, n, 3)
        omega_i = omega.T[np.newaxis, :, :]  # (1, n, 3)

        H_linear = np.cross(omega_j, v_i)  # (n, n, 3)
        H_angular = np.cross(omega_j, omega_i)  # (n, n, 3)

        # Zero out lower triangle (j > i) - we only compute for j <= i
        mask_upper = np.triu(np.ones((n, n), dtype=bool))
        H_linear = H_linear * mask_upper[:, :, np.newaxis]
        H_angular = H_angular * mask_upper[:, :, np.newaxis]

        # Apply symmetry for linear part: H_linear[i, j, :] = H_linear[j, i, :]
        i_lower, j_lower = np.tril_indices(n, k=-1)
        H_linear[i_lower, j_lower, :] = H_linear[j_lower, i_lower, :]

        # Build Hessian tensor: H[j, :, i] with shape (n, 6, n)
        H = np.zeros((n, 6, n))
        H[:, :3, :] = H_linear.transpose(0, 2, 1)  # (n, 3, n)
        H[:, 3:, :] = H_angular.transpose(0, 2, 1)  # (n, 3, n)

        # We finally mask the Hessian with the task dependent mask, if any
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
            where n is the number of degrees of freedom (model.nv).
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

            e(q) = - \dot{m}_\text{desired}\in \mathbb{R}

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
