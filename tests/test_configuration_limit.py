#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Test configuration limit."""

import unittest

import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.limits import ConfigurationLimit


class TestConfigurationLimit(unittest.TestCase):
    """Test fixture for configuration limit."""

    def setUp(self):
        """Set test fixture up."""
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        model = robot.model
        self.data = robot.data
        self.limit = ConfigurationLimit(model)
        self.model = model
        self.robot = robot

    def test_dimensions(self):
        """Check dimensions of configuration limit projection."""
        for joint in self.limit.joints:
            self.assertGreaterEqual(joint.idx_q, 0)
            self.assertGreaterEqual(joint.idx_v, 0)
        nb = len(self.limit.joints)  # those are only revolute joints
        nv = self.model.nv
        self.assertEqual(len(self.limit.indices), nb)
        self.assertEqual(self.limit.projection_matrix.shape, (nb, nv))

    def test_model_with_no_limit(self):
        """Check that unbounded models don't fail."""
        empty_model = pin.Model()
        empty_bounded = ConfigurationLimit(empty_model)
        self.assertEqual(len(empty_bounded.indices), 0)

    def test_far_from_limit(self, tol=1e-10):
        """Limit has no effect when the configuration is far away.

        Args:
            tol: Numerical tolerance.

        When we are far away from configuration limits, the velocity limit is
        simply the configuration-agnostic one from the robot.

        Note:
            This test works with a commit-pinned description of Upkie. It will
            not work for any description, as a robot may have e.g. some
            velocity-unbounded joints.
        """
        robot = load_robot_description(
            "upkie_description",
            root_joint=pin.JointModelFreeFlyer(),
            commit="62f3ba24c2045b44faedb7c6c6167e74e157b49e",
        )
        dt = 1e-3  # [s]
        configuration = Configuration(robot.model, robot.data, robot.q0)
        limit = ConfigurationLimit(robot.model)
        G, h = limit.compute_qp_inequalities(robot.q0, dt=dt)
        v_lim = configuration.model.velocityLimit
        self.assertLess(np.max(+G @ v_lim * dt - h), -tol)
        self.assertLess(np.max(-G @ v_lim * dt - h), -tol)

    def test_configuration_limit_repulsion(self, tol=1e-10):
        """Velocities are scaled down when close to a configuration limit.

        Args:
            tol: Numerical tolerance.
        """
        robot = load_robot_description(
            "upkie_description", root_joint=pin.JointModelFreeFlyer()
        )
        dt = 1e-3  # [s]
        configuration = Configuration(robot.model, robot.data, robot.q0)
        slack_vel = 5.5e-4  # [rad] / [s]

        # Clamp configuration limit to q +/- slack_vel * dt
        robot.model.lowerPositionLimit = configuration.integrate(
            -slack_vel * configuration.tangent.ones, dt
        )
        robot.model.upperPositionLimit = configuration.integrate(
            +slack_vel * configuration.tangent.ones, dt
        )

        limit = ConfigurationLimit(robot.model, config_limit_gain=0.5)
        G, h = limit.compute_qp_inequalities(configuration.q, dt)
        self.assertLess(np.max(h), slack_vel * dt + tol)
        self.assertGreater(np.min(h), -slack_vel * dt - tol)

    def test_gen2_description(self):
        robot = load_robot_description("gen2_description")
        limit = ConfigurationLimit(robot.model)
        self.assertEqual(tuple(limit.indices), (1, 2, 4))
