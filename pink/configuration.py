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

"""
Pink adds a notion of *configured* robot to indicate that forward kinematics
functions have been run, so that frame transforms and frame Jacobians used for
IK can be queried from the robot's data.
"""

import numpy as np
import pinocchio as pin


class Configuration:

    """
    Type annotation indicating that quantities that depend on the configuration
    :math:`q` have been computed.

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
        data: Data with kinematics matching the configuration vector
            :data:`Configuration.q`.
        model: Kinodynamic model.
        q: Configuration vector matching the kinematics in
            :data:`Configuration.data`.
    """

    data: pin.Data
    model: pin.Model
    q: np.ndarray

    def __init__(self, model: pin.Model, data: pin.Data, q: np.ndarray):
        self.data = data
        self.model = model
        self.q = q

    def check_limits(self):
        """
        TODO(scaron): implement.
        """
        raise NotImplementedError

    def get_body_jacobian(self, body: str) -> np.ndarray:
        """
        Compute the Jacobian matrix :math:`{}_B J_{WB}` of the body velocity
        :math:`{}_B v_{WB}`:

        .. math::

            {}_B v_{WB} = {}_B J_{WB} \\dot{q}

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
        body_id = self.model.getBodyId(body)
        J: np.ndarray = pin.getFrameJacobian(
            self.model, self.data, body_id, pin.ReferenceFrame.LOCAL
        )
        return J

    def get_transform_body_to_world(self, body: str) -> pin.SE3:
        """
        Get the pose of a body frame of the robot in its current configuration.

        Args:
            body: Body frame name, typically the link name from the URDF.

        Returns:
            Current transform from body frame to world frame.

        Raises:
            KeyError: if the body name is not found in the robot model.
        """
        body_id = self.model.getBodyId(body)
        try:
            return self.data.oMf[body_id].copy()
        except IndexError as index_error:
            raise KeyError(
                f'Body "{body}" not found in robot model'
            ) from index_error

    def integrate(self, velocity, dt) -> np.ndarray:
        """
        Integrate a velocity starting from the current configuration.

        Args:
            velocity: Velocity in tangent space.
            dt: Integration duration in [s].

        Returns:
            New configuration after integration.
        """
        return pin.integrate(self.model, self.q, velocity * dt)


def assume_configuration(
    robot: pin.RobotWrapper, q: np.ndarray
) -> Configuration:
    """
    Assume that the provided robot wrapper has already been configured.

    Args:
        robot: Robot wrapper with its initial data.
        q: Configuration matching the robot wrapper's data.

    Returns:
        Robot configuration.
    """
    return Configuration(robot.model, robot.data, q)


def apply_configuration(
    robot: pin.RobotWrapper, q: np.ndarray
) -> Configuration:
    """
    Run forward kinematics on a robot wrapper.

    Args:
        robot: Robot wrapper with its initial data.
        q: Configuration vector to apply.

    Returns:
        Configured robot.
    """
    pin.computeJointJacobians(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)
    return Configuration(robot.model, robot.data, q)
