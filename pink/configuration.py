#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Configuration space of a robot model.

Pink uses `Pinocchio <https://github.com/stack-of-tasks/pinocchio>`__ for
forward kinematics. A :class:`Configuration` is a pair of Pinocchio model and
data where forward kinematics have been run, indicating that frame transforms
and frame Jacobians used for IK can be queried.
"""

from typing import Optional

import numpy as np
import pinocchio as pin

from .exceptions import FrameNotFound, NotWithinConfigurationLimits
from .limits import ConfigurationLimit, VelocityLimit
from .utils import VectorSpace, get_root_joint_dim


class Configuration:
    """Type indicating that configuration-dependent quantities are available.

    In Pink, this type enables access to frame transforms and frame Jacobians.
    We rely on typing to make sure the proper forward kinematics functions have
    been called beforehand. In Pinocchio, these functions are:

    - ``pin.computeJointJacobians(model, data, configuration)``
    - ``pin.updateFramePlacements(model, data)``

    The former computes the full model Jacobian into ``data.J``. (It also
    computes forward kinematics, so there is no need to further call
    ``pin.forwardKinematics(model, data, configuration)``.) The latter updates
    frame placements.

    Notes:
        This class is meant to be used as a subclass of pin.RobotWrapper, not
        wrap it. However, right now pin.RobotWrapper does not have a shallow
        copy constructor. TODO(scaron): bring it up upstream.

    Attributes:
        data: Data corresponding to :data:`Configuration.model`.
        model: Kinodynamic model.
        q: Configuration vector for the robot model.
    """

    data: pin.Data
    model: pin.Model
    q: np.ndarray

    def __init__(
        self,
        model: pin.Model,
        data: pin.Data,
        q: np.ndarray,
        copy_data: bool = True,
        forward_kinematics: bool = True,
    ):
        """Initialize configuration.

        Args:
            model: Kinodynamic model.
            data: Data corresponding to the model.
            q: Configuration vector.
            copy_data: If true (default), work on an internal copy of the input
                data. Otherwise, work on the input data directly.
            forward_kinematics: If true (default), compute forward kinematics
                from the q into the internal data.

        Notes:
            Configurations copy data and run forward kinematics by default so
            that they are less error-prone for newcomers. You can avoid copies
            or forward kinematics (e.g. if it is already computed by the
            caller) using constructor parameters.
        """
        if not hasattr(model, "tangent"):
            model.tangent = VectorSpace(model.nv)
        if not hasattr(model, "configuration_limit"):
            model.configuration_limit = ConfigurationLimit(model)
        if not hasattr(model, "velocity_limit"):
            model.velocity_limit = VelocityLimit(model)
        q_readonly = q.copy()
        q_readonly.setflags(write=False)
        self.data = data.copy() if copy_data else data
        self.model = model
        self.q = q_readonly
        self.tangent = model.tangent
        #
        if forward_kinematics:
            self.update(None)

    def update(self, q: Optional[np.ndarray] = None) -> None:
        """Update configuration to a new vector and run forward kinematics.

        Args:
            q: New configuration vector.
        """
        if q is not None:
            q_readonly = q.copy()
            q_readonly.setflags(write=False)
            self.q = q_readonly
        pin.computeJointJacobians(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def check_limits(self, tol: float = 1e-6) -> None:
        """Check that the current configuration is within limits.

        Args:
            tol: Tolerance in radians.

        Raises:
            NotWithinConfigurationLimits: if the current configuration is
                within limits.
        """
        q_max = self.model.upperPositionLimit
        q_min = self.model.lowerPositionLimit
        root_nq, _ = get_root_joint_dim(self.model)
        for i in range(root_nq, self.model.nq):
            if q_max[i] <= q_min[i] + tol:  # no limit
                continue
            if self.q[i] < q_min[i] - tol or self.q[i] > q_max[i] + tol:
                raise NotWithinConfigurationLimits(
                    i,
                    self.q[i],
                    q_min[i],
                    q_max[i],
                )

    def get_frame_jacobian(self, frame: str) -> np.ndarray:
        r"""Compute the Jacobian matrix of a frame velocity.

        Denoting our frame by :math:`B` and the world frame by :math:`W`, the
        Jacobian matrix :math:`{}_B J_{WB}` is related to the body velocity
        :math:`{}_B v_{WB}` by:

        .. math::

            {}_B v_{WB} = {}_B J_{WB} \dot{q}

        Args:
            frame: Name of the frame, typically a link name from the URDF.

        Returns:
            Jacobian :math:`{}_B J_{WB}` of the frame.

        When the robot model includes a floating base
        (pin.JointModelFreeFlyer), the configuration vector :math:`q` consists
        of:

        - ``q[0:3]``: position in [m] of the floating base in the inertial
          frame, formatted as :math:`[p_x, p_y, p_z]`.
        - ``q[3:7]``: unit quaternion for the orientation of the floating base
          in the inertial frame, formatted as :math:`[q_x, q_y, q_z, q_w]`.
        - ``q[7:]``: joint angles in [rad].
        """
        if not self.model.existFrame(frame):
            raise FrameNotFound(frame, self.model.frames)
        frame_id = self.model.getFrameId(frame)
        J: np.ndarray = pin.getFrameJacobian(
            self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL
        )
        return J

    def get_transform_frame_to_world(self, frame: str) -> pin.SE3:
        """Get the pose of a frame in the current configuration.

        Args:
            frame: Name of a frame, typically a link name from the URDF.

        Returns:
            Current transform from the given frame to the world frame.

        Raises:
            KeyError: if the frame name is not found in the robot model.
        """
        frame_id = self.model.getFrameId(frame)
        try:
            return self.data.oMf[frame_id].copy()
        except IndexError as index_error:
            raise FrameNotFound(frame, self.model.frames) from index_error

    def get_transform(self, source: str, dest: str) -> pin.SE3:
        """Get the pose of a frame with respect to another frame.

        Args:
            source: Name of the frame to get the pose of.
            dest: Name of the frame to get the pose in.

        Returns:
            Current transform from the source frame to the dest frame.

        Raises:
            KeyError: if any of the frame names is not found in the model.
        """
        transform_source_to_world = self.get_transform_frame_to_world(source)
        transform_dest_to_world = self.get_transform_frame_to_world(dest)
        return transform_dest_to_world.actInv(transform_source_to_world)

    def integrate(self, velocity, dt) -> np.ndarray:
        """Integrate a velocity starting from the current configuration.

        Args:
            velocity: Velocity in tangent space.
            dt: Integration duration in [s].

        Returns:
            New configuration vector after integration.
        """
        return pin.integrate(self.model, self.q, velocity * dt)

    def integrate_inplace(self, velocity, dt) -> None:
        """Integrate a velocity starting from the current configuration.

        Args:
            velocity: Velocity in tangent space.
            dt: Integration duration in [s].
        """
        q = pin.integrate(self.model, self.q, velocity * dt)
        self.update(q)
