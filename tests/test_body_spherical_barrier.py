#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Ivan Domrachev, Simeon Nedelchev

"""Tests for position barrier limit."""

import unittest

import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description

from pink import Configuration
from pink.barriers import BodySphericalBarrier
from pink.exceptions import NegativeMinimumDistance


class TestBodySphericalBarrier(unittest.TestCase):
    """Testing position barrier limit."""

    def setUp(self):
        """Set test fixture up."""
        robot = load_robot_description("yumi_description")
        model = robot.model
        self.data = robot.data
        self.model = model
        self.robot = robot
        self.ees = ("yumi_link_7_l", "yumi_link_7_r")
        self.q0 = np.array(
            [
                0.045,
                -0.155,
                -0.394,
                -0.617,
                -0.939,
                -0.343,
                -1.216,
                0,
                0,
                -0.374,
                -0.249,
                0.562,
                -0.520,
                0.934,
                -0.337,
                1.400,
                0,
                0,
            ]
        )
        self.configuration = Configuration(self.model, self.data, self.q0)

    def test_negative_distance(self):
        """Raise an error when the robot body is not found."""
        with self.assertRaises(NegativeMinimumDistance):
            BodySphericalBarrier(self.ees, d_min=-1)

    def test_dimension(self):
        """Check dimensions of configuration limit projection."""
        self.assertEqual(BodySphericalBarrier(self.ees, d_min=0.2).dim, 1)

    def test_gains(self):
        """Check gains of configuration limit projection."""
        # One limit, scalar gain
        self.assertEqual(
            BodySphericalBarrier(self.ees, d_min=0.2).gain.shape, (1,)
        )

    def test_jacobians(self):
        """Test that shapes of jacobians in all barriers are correct."""

        barrier = BodySphericalBarrier(self.ees, d_min=0.2)
        J = barrier.compute_jacobian(self.configuration)
        self.assertEqual(J.ndim, 1)
        self.assertEqual(J.shape[0], self.robot.nv)

    def test_positive_when_in_safety_zone(self):
        """Check that the barrier is positive when in the safety zone."""

        barrier = BodySphericalBarrier(self.ees, d_min=0.2)
        h = barrier.compute_barrier(self.configuration)
        self.assertTrue(h[0] > 0)

    def test_negative_when_out_of_safety_zone(self):
        """Test that the barrier is negative when out of the safety zone."""

        barrier = BodySphericalBarrier(self.ees, d_min=0.3)
        h = barrier.compute_barrier(self.configuration)
        self.assertTrue(h[0] < 0)
