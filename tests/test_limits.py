#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

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

    def test_limit_dimension(self):
        """Velocity limit dimension is the number of bounded joints."""
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        model = robot.model
        dt = 1e-3  # [s]
        for limit in [ConfigurationLimit(model), VelocityLimit(model)]:
            G, h = limit.compute_qp_inequalities(robot.q0, dt)
            self.assertEqual(G.shape[0], h.shape[0])
            self.assertEqual(G.shape[1], model.nv)

    def test_model_with_no_joint_limit(self):
        """Model with no joint limit has no velocity-limit vector."""
        model = pin.Model()
        model.addJoint(
            0, pin.JointModelSpherical(), pin.SE3.Identity(), "spherical"
        )
        robot = pin.RobotWrapper(model=model)
        dt = 1e-3  # [s]
        for limit in [ConfigurationLimit(model), VelocityLimit(model)]:
            return_value = limit.compute_qp_inequalities(robot.q0, dt)
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
        for limit in [ConfigurationLimit(model), VelocityLimit(model)]:
            return_value = limit.compute_qp_inequalities(robot.q0, dt)
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
