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

"""Tests that should pass for all limits."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.limits import ConfigurationLimit, VelocityLimit
from pink.solve_ik import build_ik


class TestLimits(unittest.TestCase):
    """Tests that should pass for all limits."""

    def setUp(self):
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        limits = [
            ConfigurationLimit(robot.model),
            VelocityLimit(robot.model),
        ]
        self.data = robot.data
        self.limits = limits
        self.model = robot.model
        self.robot = robot

    def test_limit_dimension(self):
        """Velocity limit dimension is the number of bounded joints."""
        dt = 1e-3  # [s]
        for limit in self.limits:
            G, h = limit.compute_qp_inequalities(self.model, self.robot.q0, dt)
            self.assertEqual(G.shape[0], h.shape[0])
            self.assertEqual(G.shape[1], (self.model.bounded_tangent.dim,))

    def test_model_with_no_joint_limit(self):
        """Model with no joint limit has no velocity-limit vector."""
        model = pin.Model()
        model.addJoint(
            0, pin.JointModelSpherical(), pin.SE3.Identity(), "spherical"
        )
        robot = pin.RobotWrapper(model=model)
        dt = 1e-3  # [s]
        for limit in self.limits:
            return_value = limit.compute_qp_inequalities(model, robot.q0, dt)
            self.assertIsNone(return_value)

    def test_model_with_limitless_joint(self):
        """Same as previous test, but the joint has a limit set to zero."""
        model = pin.Model()
        model.addJoint(
            0,
            pin.JointModelRevoluteUnaligned(),
            pin.SE3.Identity(),
            "revolute",
            max_effort=np.array([0.0]),
            max_velocity=np.array([0.0]),
            min_config=np.array([0.0]),
            max_config=np.array([0.0]),
        )
        robot = pin.RobotWrapper(model=model)
        dt = 1e-3  # [s]
        for limit in self.limits:
            return_value = limit.compute_qp_inequalities(model, robot.q0, dt)
            self.assertIsNone(return_value)

    def test_velocity_without_configuration_limits(self, tol: float = 1e-10):
        """Velocity limits are loaded for a model without config limits.

        Args:
            tol: Equality test tolerance.

        Note:
            The loaded robot description is commit-pinned to make sure we load
            a URDF without configuration limits.
        """
        dt = 1e-3  # [s]
        sigmaban = load_robot_description(
            "sigmaban_description",
            commit="d5d023fd35800d00d7647000bce8602617a4960d",
        )
        configuration = Configuration(
            sigmaban.model, sigmaban.data, sigmaban.q0
        )
        problem = build_ik(configuration, [], dt)
        G, h = problem.G, problem.h
        self.assertIsNotNone(G)
        self.assertIsNotNone(h)
        self.assertLess(np.max(h - sigmaban.model.velocityLimit), tol)
