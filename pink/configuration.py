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

"""Configuration type.

Pink uses `Pinocchio <https://github.com/stack-of-tasks/pinocchio>`__ for
forward kinematics. It adds to it a :class:`Configuration` type to indicate
that forward kinematics functions have been run, indicating that frame
transforms and frame Jacobians used for IK can be queried from the robot's
data.
"""

import numpy as np
import pinocchio as pin

from .exceptions import BodyNotFound, NotWithinConfigurationLimits
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
        copy_data=True,
        forward_kinematics=True,
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
            self.update()

    def update(self) -> None:
        """Run forward kinematics from the configuration."""
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

    def get_frame_jacobian(self, body: str) -> np.ndarray:
        r"""Compute the Jacobian matrix of the body velocity.

        This matrix :math:`{}_B J_{WB}` is related to the body velocity
        :math:`{}_B v_{WB}` by:

        .. math::

            {}_B v_{WB} = {}_B J_{WB} \dot{q}

        Args:
            body: Body frame name, typically the link name from the URDF.

        Returns:
            Jacobian :math:`{}_B J_{WB}` of the body twist.

        When the robot model includes a floating base
        (pin.JointModelFreeFlyer), the configuration vector :math:`q` consists
        of:

        - ``q[0:3]``: position in [m] of the floating base in the inertial
          frame, formatted as :math:`[p_x, p_y, p_z]`.
        - ``q[3:7]``: unit quaternion for the orientation of the floating base
          in the inertial frame, formatted as :math:`[q_x, q_y, q_z, q_w]`.
        - ``q[7:]``: joint angles in [rad].
        """
        if not self.model.existFrame(body):
            raise BodyNotFound(f"body {body} does not exist")
        body_id = self.model.getFrameId(body)
        J: np.ndarray = pin.getFrameJacobian(
            self.model, self.data, body_id, pin.ReferenceFrame.LOCAL
        )
        return J

    def get_transform_frame_to_world(self, body: str) -> pin.SE3:
        """Get the pose of a body frame in the current configuration.

        Args:
            body: Body frame name, typically the link name from the URDF.

        Returns:
            Current transform from body frame to world frame.

        Raises:
            KeyError: if the body name is not found in the robot model.
        """
        body_id = self.model.getFrameId(body)
        try:
            return self.data.oMf[body_id].copy()
        except IndexError as index_error:
            raise KeyError(
                f'Body "{body}" not found in robot model'
            ) from index_error

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
        self.q = pin.integrate(self.model, self.q, velocity * dt)
        self.update()
